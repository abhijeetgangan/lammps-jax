"""Record plan-compile calls of an FFI kernel library so a shim can reissue them.

Wraps any ctypes function taking int64 scalars, byte strings, and int64
pointer/length pairs; the shim replays the calls through libffi at run time.
"""
import ctypes
import struct

TAG_SCALAR, TAG_BYTES, TAG_ARRAY = 0, 1, 2

RECORDS = []


def install(library, compile_symbol):
    """Wrap library.<compile_symbol> to record each call; returns the original.

    Pointer arguments must be followed by their int64 length argument.
    Installing twice keeps the first wrapper.
    """
    real = getattr(library, compile_symbol)
    if getattr(real, "ffi_replay_real", None) is not None:
        return real.ffi_replay_real
    argtypes = list(real.argtypes)

    def wrapper(*args):
        recorded = []
        i = 0
        while i < len(args):
            argtype = argtypes[i] if i < len(argtypes) else None
            value = args[i]
            if argtype is ctypes.c_char_p or isinstance(value, (bytes, bytearray)):
                recorded.append((TAG_BYTES, bytes(value)))
                i += 1
            elif hasattr(argtype, "_type_") and not isinstance(value, int):
                length = int(args[i + 1])
                recorded.append((TAG_ARRAY, [int(value[k]) for k in range(length)]))
                i += 2
            else:
                recorded.append((TAG_SCALAR, int(value)))
                i += 1
        handle = real(*args)
        RECORDS.append((recorded, int(handle)))
        return handle

    wrapper.ffi_replay_real = real
    setattr(library, compile_symbol, wrapper)
    return real


def dump(path, compile_symbol, execute_symbol):
    """Write RECORDS plus the two symbol names to path; returns the record count.

    Record one bundle per process: handles are registry indices and the
    registry must start fresh for replay to reproduce them.
    """

    def packed_bytes(data):
        return struct.pack("<Q", len(data)) + data

    out = [struct.pack("<8s", b"FFIRPLY1"),
           packed_bytes(compile_symbol.encode()),
           packed_bytes(execute_symbol.encode()),
           struct.pack("<Q", len(RECORDS))]
    for recorded, handle in RECORDS:
        out.append(struct.pack("<qQ", handle, len(recorded)))
        for tag, payload in recorded:
            out.append(struct.pack("<B", tag))
            if tag == TAG_SCALAR:
                out.append(struct.pack("<q", payload))
            elif tag == TAG_BYTES:
                out.append(packed_bytes(payload))
            else:
                out.append(struct.pack("<Q", len(payload)))
                out.append(struct.pack(f"<{len(payload)}q", *payload) if payload else b"")
    with open(path, "wb") as handle_out:
        handle_out.write(b"".join(out))
    return len(RECORDS)
