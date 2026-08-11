"""Export DYNAMO setfl (eam/alloy) tables as lammps-jax bundles.

Set POTENTIAL and the capacities, then run `julia --project=. eam_export.jl`.
LAMMPS type t carries element t of the file; examples/in.eam_cuzr runs the
bundle through its backend switch.
"""

using LinearAlgebra
using Reactant, Enzyme

include(joinpath(@__DIR__, "bundle_utils.jl"))

POTENTIAL = joinpath(@__DIR__, "..", "potentials", "CuZr.eam.alloy.gz")
OUTPUT = "cuzr.lammps-jax.json"
MAX_ATOMS = 3072
EDGES_PER_ATOM = 16
MAX_EDGES = MAX_ATOMS * EDGES_PER_ATOM
N_HOPS = 1
SKIP_CHECK = false
# python lammps_jax.eam reference on the parity cluster below.
REF_E = -228.9739846203297
REF_MAXF = 1.6237205785541458

read_table(path) = endswith(path, ".gz") ?
    read(pipeline(`gzip -dc $path`), String) : read(path, String)

# PairEAM::interpolate: value(t) = c1 + t*(c2 + t*(c3 + t*c4)) per interval.
function spline_coefficients(f)
    n = length(f)
    d = similar(f)
    d[1] = f[2] - f[1]
    d[2] = 0.5 * (f[3] - f[1])
    d[n - 1] = 0.5 * (f[n] - f[n - 2])
    d[n] = f[n] - f[n - 1]
    for m in 3:n - 2
        d[m] = ((f[m - 2] - f[m + 2]) + 8.0 * (f[m + 1] - f[m - 1])) / 12.0
    end
    step = diff(f)
    c2 = zeros(n)
    c3 = zeros(n)
    c2[1:n - 1] = 3.0 .* step .- 2.0 .* d[1:n - 1] .- d[2:n]
    c3[1:n - 1] = d[1:n - 1] .+ d[2:n] .- 2.0 .* step
    return hcat(f, d, c2, c3)
end

function load_setfl(path)
    tokens = split(join(split(read_table(path), "\n")[4:end], " "))
    ne = parse(Int, tokens[1])
    elements = tokens[2:1 + ne]
    p = 2 + ne
    nrho, nr = parse(Int, tokens[p]), parse(Int, tokens[p + 2])
    drho, dr = parse(Float64, tokens[p + 1]), parse(Float64, tokens[p + 3])
    cutoff = parse(Float64, tokens[p + 4])
    p += 5
    emb, dens = Matrix{Float64}[], Matrix{Float64}[]
    for _ in 1:ne
        p += 4
        push!(emb, spline_coefficients(parse.(Float64, tokens[p:p + nrho - 1])))
        p += nrho
        push!(dens, spline_coefficients(parse.(Float64, tokens[p:p + nr - 1])))
        p += nr
    end
    pair = Matrix{Float64}[]
    pair_index = zeros(Int32, ne, ne)
    for i in 1:ne, j in 1:i
        pair_index[i, j] = pair_index[j, i] = length(pair)
        push!(pair, spline_coefficients(parse.(Float64, tokens[p:p + nr - 1])))
        p += nr
    end
    @assert p == length(tokens) + 1 "unparsed trailing values"
    return (; elements, nrho, drho, nr, dr, cutoff,
            embedding = vcat(emb...), density = vcat(dens...),
            pair = vcat(pair...),
            pair_flat = Int32.(vec(permutedims(pair_index))))
end

# Stacked (table*n + node): one gather serves every element.
function spline_lookup(flat, ids0, x, delta, n; extrapolate = false)
    idx = x ./ delta
    node_i = Int32.(floor.(idx))
    node = clamp.(node_i, Int32(0), Int32(n - 2))
    t_raw = idx .- node
    t = min.(t_raw, 1.0)
    c = flat[ids0 .* Int32(n) .+ node .+ Int32(1), :]
    v = c[:, 1] .+ t .* (c[:, 2] .+ t .* (c[:, 3] .+ t .* c[:, 4]))
    if extrapolate
        v = v .+ (t_raw .- t) .* (c[:, 2] .+ 2.0 .* c[:, 3] .+ 3.0 .* c[:, 4])
    end
    return v
end

function site_energies(positions, species0, centers, neighbors, n_atoms, tb, em)
    rij = positions[:, neighbors] .- positions[:, centers]
    r_sq = sum(abs2, rij; dims = 1)[1, :]
    vf = Float64.(r_sq .< T.cutoff^2)
    if em !== nothing
        vf = Float64.(em) .* vf
    end
    r = sqrt.(vf .* r_sq .+ (1.0 .- vf))
    snd = species0[centers]
    rcv = species0[neighbors]

    rho = vf .* spline_lookup(tb.density, rcv, r, T.dr, T.nr)
    density = segment_sum(reshape(rho, :, 1), centers, n_atoms)[:, 1]
    pid = tb.pair_flat[snd .* Int32(length(T.elements)) .+ rcv .+ Int32(1)]
    z2 = spline_lookup(tb.pair, pid, r, T.dr, T.nr)
    pair_term = vf .* (0.5 .* z2 ./ r)
    pair_energy = segment_sum(reshape(pair_term, :, 1), centers, n_atoms)[:, 1]
    embed = spline_lookup(tb.embedding, species0, density, T.drho, T.nrho;
                          extrapolate = true)
    return pair_energy .+ embed
end

function core_energy(positions, species, nlocal, nghost, senders, receivers,
                     edge_mask)
    c = Reactant.Ops.constant
    tb = (embedding = c(T.embedding), density = c(T.density), pair = c(T.pair),
          pair_flat = c(T.pair_flat))
    em_i = Int32.(edge_mask)
    centers = em_i .* (senders .+ Int32(1)) .+ (Int32(1) .- em_i)
    neighbors = em_i .* (receivers .+ Int32(1)) .+ (Int32(1) .- em_i)
    spec = clamp.(species, Int32(0), Int32(length(T.elements) - 1))
    site = site_energies(positions, spec, centers, neighbors, MAX_ATOMS, tb,
                         edge_mask)
    rows = c(collect(Int32, 0:MAX_ATOMS - 1))
    nl, ng = nlocal[], nghost[]
    local_e = sum(Float64.(rows .< nl) .* site)
    total_e = sum(Float64.(rows .< (nl + ng)) .* site)
    N_HOPS > 1 && return local_e
    ghost = min(1.0, sum(Float64.(edge_mask .& (senders .>= nl))))
    return ghost * total_e + (1.0 - ghost) * local_e
end

bundle_energy(p, s, nl, ng, sd, rc, em) =
    core_energy(p, s, nl, ng, sd, rc, em) + anchor(p, s, nl, ng, sd, rc, em)

function bundle_forces(p, s, nl, ng, sd, rc, em)
    g = Enzyme.gradient(Reverse, Const(core_energy), p, Const(s), Const(nl),
                        Const(ng), Const(sd), Const(rc), Const(em))[1]
    return (-1.0) .* g .+ 0.0 .* anchor(s, nl, ng, sd, rc, em)
end

bundle_energy_and_forces(p, s, nl, ng, sd, rc, em) =
    (bundle_energy(p, s, nl, ng, sd, rc, em),
     bundle_forces(p, s, nl, ng, sd, rc, em))


T = load_setfl(POTENTIAL)
println("$(POTENTIAL): elements $(T.elements) cutoff $(T.cutoff)")

a0 = 3.26
pos, spec = Vector{Float64}[], Int32[]
for i in 0:2, j in 0:2, k in 0:2
    push!(pos, a0 .* [i, j, k]); push!(spec, 0)
    push!(pos, a0 .* [i + 0.5, j + 0.5, k + 0.5]); push!(spec, 1)
end
positions = reduce(hcat, pos)
positions[:, 1] += [0.1, 0.07, -0.05]
n_check = length(spec)
edges = [(i, j) for i in 1:n_check for j in 1:n_check
         if i != j && norm(positions[:, i] - positions[:, j]) < T.cutoff]
tb_plain = (embedding = T.embedding, density = T.density, pair = T.pair,
            pair_flat = T.pair_flat)
site = site_energies(positions, spec, Int32.(first.(edges)),
                     Int32.(last.(edges)), n_check, tb_plain, nothing)
energy = sum(site)
println("parity vs python: E $REF_E vs $energy (d $(abs(energy - REF_E)))")
@assert abs(energy - REF_E) < 1e-12 * abs(REF_E)

# The parity cluster padded to capacity is the example input.
args = (Reactant.to_rarray([positions zeros(3, MAX_ATOMS - n_check)]),
        Reactant.to_rarray([spec; zeros(Int32, MAX_ATOMS - n_check)]),
        Reactant.to_rarray(fill(Int32(n_check))),
        Reactant.to_rarray(fill(Int32(0))),
        Reactant.to_rarray(Int32.([first.(edges) .- 1;
                                   fill(MAX_ATOMS, MAX_EDGES - length(edges))])),
        Reactant.to_rarray(Int32.([last.(edges) .- 1;
                                   fill(MAX_ATOMS, MAX_EDGES - length(edges))])),
        Reactant.to_rarray([ones(Bool, length(edges));
                            zeros(Bool, MAX_EDGES - length(edges))]))

if !SKIP_CHECK
    compiled = @compile bundle_energy_and_forces(args...)
    e_c, f_c = compiled(args...)
    max_f = maximum(abs.(Array(f_c)[:, 1:n_check]))
    println("compiled parity: E $(Float64(e_c)) max|F| $max_f (ref $REF_MAXF)")
    @assert abs(Float64(e_c) - REF_E) < 1e-12 * abs(REF_E)
    @assert abs(max_f - REF_MAXF) < 1e-12 * REF_MAXF
end

programs = Dict(
    "force_mlir_b64" =>
        program_b64(bundle_forces, args, MAX_ATOMS, MAX_EDGES, "force"),
    "energy_mlir_b64" =>
        program_b64(bundle_energy, args, MAX_ATOMS, MAX_EDGES, "energy"),
    "energy_and_forces_mlir_b64" =>
        program_b64(bundle_energy_and_forces, args, MAX_ATOMS, MAX_EDGES,
                    "energy_and_forces"))
write_bundle(OUTPUT, programs, Dict(
    "input_layout" => "sparse-edge",
    "max_atoms" => MAX_ATOMS,
    "max_edges" => MAX_EDGES,
    "cutoff" => T.cutoff,
    "unit_style" => "metal",
    "precision" => "float64",
    "force_output" => "atom-force",
    "newton" => "on",
    "n_hops" => N_HOPS,
    "edge_pairing" => "full",
    "comm_widths" => Int[],
    "custom_call_targets" => String[],
    "uses_box" => false,
    "n_species" => length(T.elements)))
