# Shared bundle plumbing for the Julia exporters.

using Base64, Reactant

Reactant.set_default_backend("cpu")

TINY = floatmin(Float64)
# xla.CompileOptions: device_ordinal -1, 1 replica, 1 partition, portable; matches export.py.
COMPILE_OPTIONS_B64 = base64encode(UInt8[
    0x1a, 0x19, 0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x01, 0x20, 0x01, 0x28, 0x01, 0x62, 0x01, 0x00, 0x92, 0x01, 0x01, 0x00,
    0xb8, 0x01, 0x01, 0x20, 0x01,
])

# Reactant ships no traced Bool -> Float64 conversion; select does the job.
Base.Float64(x::Reactant.TracedRNumber{Bool}) = ifelse(x, 1.0, 0.0)

segment_sum(rows, idx, n) = permutedims(idx .== permutedims(1:n)) * rows

# jax's abi_anchor, added outside the gradient (what stop_gradient does there).
anchor(vals...) = sum(vals) do v
    TINY * (v isa AbstractArray{<:Any, 0} ? 1.0 * v[] :
            eltype(v) == Bool ? sum(Float64.(v)) : sum(1.0 .* v))
end

function program_b64(f, args, max_atoms, max_edges, name)
    mod = @code_hlo f(args...)
    # The plugin parses pure StableHLO; drop Reactant's enzymexla annotations.
    text = replace(string(mod), r"\s*\{enzymexla\.[^}]*\}" => "")
    text = replace(text, r" attributes \{$"m => " {")
    # Julia dims trace reversed: positions must enter as (3, max_atoms) columns.
    header = split(text, "\n")[2]
    for shape in ("$(max_atoms)x3xf64", "$(max_edges)xi1")
        occursin("tensor<$shape>", header) ||
            error("$name signature lost tensor<$shape>")
    end
    println("$name: $(length(text)) chars of StableHLO")
    return base64encode(text)
end

# Sorted keys match export.py's writer; the C++ first-match scanner assumes it.
function write_json(io, x, depth)
    pad = "  " ^ depth
    if x isa Dict
        print(io, "{\n")
        names = sort(collect(keys(x)))
        for (i, k) in enumerate(names)
            print(io, pad, "  \"", k, "\": ")
            write_json(io, x[k], depth + 1)
            print(io, i == length(names) ? "\n" : ",\n")
        end
        print(io, pad, "}")
    elseif x isa AbstractVector
        isempty(x) ? print(io, "[]") :
            print(io, "[", join([sprint(write_json, v, depth) for v in x], ", "), "]")
    elseif x isa AbstractString
        print(io, "\"", x, "\"")
    elseif x isa Bool || x isa Integer
        print(io, x)
    else
        print(io, Float64(x))
    end
end

function write_bundle(path, programs, contract)
    bundle = Dict(
        "format" => contract["n_hops"] > 1 ? "lammps-jax-json-distributed" :
                    "lammps-jax-json",
        "programs" => programs,
        "export_info" => Dict("exporter" => "reactant",
                              "reactant_version" => string(pkgversion(Reactant)),
                              "julia_version" => string(VERSION)),
        "compile_options_b64" => COMPILE_OPTIONS_B64,
        "contract" => contract)
    open(path, "w") do io
        write_json(io, bundle, 0)
        print(io, "\n")
    end
    println("exported ", path)
end
