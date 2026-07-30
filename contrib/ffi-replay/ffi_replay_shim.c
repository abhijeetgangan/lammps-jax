/* Replay shim for FFI kernels whose plan handles are process-local. On the
 * first invocation it reissues the recorded plan-compile calls through
 * libffi, verifies the returned handles, and forwards call frames to the
 * library's real execute handler. Slot trampolines ffi_handler_0..3 resolve
 * FFI_HANDLER_<n> specs for kernels that need no replay. Replay must stay
 * lazy: dlopening CUDA libraries under the loader lock deadlocks. See
 * README.md for the environment variables and build line.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <ffi.h>
#include <link.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_SCALAR 0
#define TAG_BYTES 1
#define TAG_ARRAY 2
#define MAX_ARGS 64

typedef void *(*handler_fn)(void *);

static handler_fn real_handler = NULL;

static void die(const char *msg) {
    fprintf(stderr, "ffi_replay_shim: FATAL: %s\n", msg);
    abort();
}

static uint64_t rd_u64(FILE *f) {
    uint64_t v;
    if (fread(&v, 8, 1, f) != 1) die("truncated replay file");
    return v;
}

static int64_t rd_i64(FILE *f) {
    int64_t v;
    if (fread(&v, 8, 1, f) != 1) die("truncated replay file");
    return v;
}

/* Length cap keeps the size arithmetic below from overflowing. */
static uint64_t rd_len(FILE *f) {
    uint64_t len = rd_u64(f);
    if (len > (1ULL << 31)) die("implausible length in replay file");
    return len;
}

static char *rd_bytes(FILE *f, uint64_t *out_len) {
    uint64_t len = rd_len(f);
    char *buf = malloc(len + 1);
    if (!buf) die("out of memory");
    if (len && fread(buf, 1, len, f) != len) die("truncated bytes");
    buf[len] = 0;
    if (out_len) *out_len = len;
    return buf;
}

static void replay(void) {
    const char *blob = getenv("FFI_REPLAY_FILE");
    const char *libpath = getenv("FFI_REPLAY_LIB");
    if (!blob || !libpath) die("FFI_REPLAY_FILE and FFI_REPLAY_LIB must be set");

    FILE *f = fopen(blob, "rb");
    if (!f) die("cannot open FFI_REPLAY_FILE");
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "FFIRPLY1", 8) != 0)
        die("bad replay file magic");
    char *compile_symbol = rd_bytes(f, NULL);
    char *execute_symbol = rd_bytes(f, NULL);
    uint64_t count = rd_u64(f);

    void *lib = dlopen(libpath, RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); die("cannot load FFI_REPLAY_LIB"); }
    void *compile = dlsym(lib, compile_symbol);
    real_handler = (handler_fn) dlsym(lib, execute_symbol);
    if (!compile || !real_handler) die("compile/execute symbol not found in FFI_REPLAY_LIB");

    for (uint64_t idx = 0; idx < count; ++idx) {
        int64_t expected = rd_i64(f);
        uint64_t nargs_recorded = rd_u64(f);

        ffi_type *types[MAX_ARGS];
        void *values[MAX_ARGS];
        int64_t scalars[MAX_ARGS];
        void *pointers[MAX_ARGS];
        void *owned[MAX_ARGS];
        unsigned nowned = 0, n = 0;

        for (uint64_t a = 0; a < nargs_recorded; ++a) {
            uint8_t tag;
            if (fread(&tag, 1, 1, f) != 1) die("truncated tag");
            if (n + 2 > MAX_ARGS) die("too many arguments in record");
            if (tag == TAG_SCALAR) {
                scalars[n] = rd_i64(f);
                types[n] = &ffi_type_sint64;
                values[n] = &scalars[n];
                n++;
            } else if (tag == TAG_BYTES) {
                char *buf = rd_bytes(f, NULL);
                owned[nowned++] = buf;
                pointers[n] = buf;
                types[n] = &ffi_type_pointer;
                values[n] = &pointers[n];
                n++;
            } else if (tag == TAG_ARRAY) {
                uint64_t len = rd_len(f);
                int64_t *arr = len ? malloc(len * 8) : NULL;
                if (len && !arr) die("out of memory");
                if (len && fread(arr, 8, len, f) != len) die("truncated array");
                if (arr) owned[nowned++] = arr;
                pointers[n] = arr;
                types[n] = &ffi_type_pointer;
                values[n] = &pointers[n];
                n++;
                scalars[n] = (int64_t) len;
                types[n] = &ffi_type_sint64;
                values[n] = &scalars[n];
                n++;
            } else {
                die("unknown argument tag");
            }
        }

        ffi_cif cif;
        if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, n, &ffi_type_sint64, types) != FFI_OK)
            die("ffi_prep_cif failed");
        int64_t handle = 0;
        ffi_call(&cif, FFI_FN(compile), &handle, values);

        if (handle != expected) {
            fprintf(stderr,
                    "ffi_replay_shim: replayed compile %llu returned handle %lld "
                    "(recorded %lld) - registry behavior changed; refusing to run "
                    "with wrong plans\n",
                    (unsigned long long) idx, (long long) handle, (long long) expected);
            abort();
        }
        for (unsigned k = 0; k < nowned; ++k) free(owned[k]);
    }
    fclose(f);
    fprintf(stderr, "ffi_replay_shim: replayed %llu plan(s) for %s\n",
            (unsigned long long) count, compile_symbol);
    free(compile_symbol);
    free(execute_symbol);
}

static pthread_once_t replay_once = PTHREAD_ONCE_INIT;

void *ffi_replay_handler(void *call_frame) {
    pthread_once(&replay_once, replay);
    return real_handler(call_frame);
}

static pthread_once_t preload_once = PTHREAD_ONCE_INIT;

static void preload(void) {
    const char *list = getenv("FFI_PRELOAD_LIBS");
    if (!list) return;
    char *copy = strdup(list);
    for (char *lib = strtok(copy, ":"); lib; lib = strtok(NULL, ":"))
        if (!dlopen(lib, RTLD_NOW | RTLD_GLOBAL)) die("cannot load a FFI_PRELOAD_LIBS entry");
    free(copy);
}

static handler_fn slot_resolve(const char *env_name) {
    pthread_once(&preload_once, preload);
    const char *spec = getenv(env_name);
    if (!spec) die("FFI_HANDLER_<n> env var must be set for this slot");
    char *copy = strdup(spec);
    char *guard = strrchr(copy, '@');
    char *sep = strrchr(copy, '+');
    void *out = NULL;
    if (sep && strncmp(sep + 1, "0x", 2) == 0) {
        *sep = 0;
        if (guard) {
            *guard = 0;
            struct stat st;
            if (stat(copy, &st) != 0 || st.st_size != (off_t) strtoull(guard + 1, NULL, 10))
                die("slot library size differs from the recorded offset's; re-export");
        }
        void *lib = dlopen(copy, RTLD_NOW | RTLD_GLOBAL);
        if (!lib) {
            fprintf(stderr, "dlopen: %s\n", dlerror());
            die("cannot load slot library");
        }
        struct link_map *lm = NULL;
        if (dlinfo(lib, RTLD_DI_LINKMAP, &lm) != 0 || !lm) die("dlinfo failed");
        out = (char *) lm->l_addr + strtoull(sep + 1, NULL, 16);
    } else {
        sep = strrchr(copy, ':');
        if (!sep) die("slot spec must be lib.so:symbol or lib.so+0xOFFSET");
        *sep = 0;
        void *lib = dlopen(copy, RTLD_NOW | RTLD_GLOBAL);
        if (!lib) {
            fprintf(stderr, "dlopen: %s\n", dlerror());
            die("cannot load slot library");
        }
        out = dlsym(lib, sep + 1);
        if (!out) die("slot symbol not found");
    }
    free(copy);
    return (handler_fn) out;
}

#define SLOT(n) \
    static handler_fn slot_fn_##n; \
    static pthread_once_t slot_once_##n = PTHREAD_ONCE_INIT; \
    static void slot_init_##n(void) { slot_fn_##n = slot_resolve("FFI_HANDLER_" #n); } \
    void *ffi_handler_##n(void *call_frame) { \
        pthread_once(&slot_once_##n, slot_init_##n); \
        return slot_fn_##n(call_frame); \
    }
SLOT(0)
SLOT(1)
SLOT(2)
SLOT(3)
