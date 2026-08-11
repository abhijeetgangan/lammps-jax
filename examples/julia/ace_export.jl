"""Export ACEpotentials models as lammps-jax bundles.

Set OUTPUT and the model block, then run `julia --project=. ace_export.jl`.
The Al deck examples/in.mlip_al runs any bundle via -var bundle.
"""

using ACEpotentials, LinearAlgebra, Random, SparseArrays, StaticArrays
import Polynomials4ML as P4ML
using Reactant, Enzyme

include(joinpath(@__DIR__, "bundle_utils.jl"))
M = ACEpotentials.Models

OUTPUT = "ace_al.lammps-jax.json"
ELEMENTS = [:Al]
ORDER = 3
TOTALDEGREE = 8
SEED = 7
NGRID = 32768
MAX_ATOMS = 2560
EDGES_PER_ATOM = 16
MAX_EDGES = MAX_ATOMS * EDGES_PER_ATOM
N_HOPS = 1
SKIP_CHECK = false


# The UPPER tables these reference are built below.

# The grid stops just inside rcut, where the spline domain ends.
function radial_table(basis, ps_b, st_b, rcut)
    grid = collect(range(0.1, rcut - 1e-8, length = NGRID))
    Rnl, dRnl = M.evaluate_ed_batched(basis, grid, z, fill(z, NGRID), ps_b, st_b)
    return grid, Matrix(Rnl), Matrix(dRnl)
end

# Repeated multiplies: Enzyme cannot reverse variable-exponent power reductions.
function monomial_col(u, e)
    col = nothing
    for k in 1:3, _ in 1:e[k]
        col = col === nothing ? u[:, k] : col .* u[:, k]
    end
    return col === nothing ? one.(u[:, 1]) : col
end
monomials(u, exps) = reduce(hcat, [monomial_col(u, e) for e in exps])

function hermite(r, grid0, h, n, values, derivs)
    x = clamp.((r .- grid0) ./ h, 0.0, n - 1.0 - 1e-9)
    i = Int32.(floor.(x))
    t = x .- i
    y0, y1 = values[i .+ 1, :], values[i .+ 2, :]
    d0, d1 = derivs[i .+ 1, :] .* h, derivs[i .+ 2, :] .* h
    h00 = @. (1 + 2t) * (1 - t)^2
    h10 = @. t * (1 - t)^2
    h01 = @. t^2 * (3 - 2t)
    h11 = @. t^2 * (t - 1)
    @. h00 * y0 + h10 * d0 + h01 * y1 + h11 * d1
end

function site_energies(positions, centers, neighbors, n_atoms, tb, em)
    vectors = positions[:, neighbors] .- positions[:, centers]
    if em !== nothing
        em_r = permutedims(Float64.(em))
        vectors = vectors .* em_r .+ [1.0, 0.0, 0.0] .* (1.0 .- em_r)
    end
    lengths = sqrt.(sum(abs2, vectors; dims = 1))[1, :]
    u = permutedims(vectors ./ permutedims(lengths))

    rnl = hermite(lengths, HR[1], HR[2], HR[3], tb.rnl, tb.drnl)
    ylm = reduce(hcat, [monomials(u, EXPONENTS[l + 1]) * tb.maps[l + 1]
                        for l in 0:LMAX])
    edge_a = rnl[:, A_NL] .* ylm[:, A_LM]
    pair_rows = hermite(lengths, HP[1], HP[2], HP[3], tb.rpair, tb.drpair)
    if em !== nothing
        edge_a = Float64.(em) .* edge_a
        pair_rows = Float64.(em) .* pair_rows
    end
    a = segment_sum(edge_a, centers, n_atoms)
    aa = reduce(hcat, [reduce(.*, [a[:, spec[:, k]] for k in 1:size(spec, 2)])
                       for spec in AA_SPECS])
    return (aa * tb.a2b') * tb.wb .+
           segment_sum(pair_rows, centers, n_atoms) * tb.wpair .+ E0
end

function core_energy(positions, species, nlocal, nghost, senders, receivers,
                     edge_mask)
    c = Reactant.Ops.constant
    tb = (rnl = c(TABLES.rnl), drnl = c(TABLES.drnl), rpair = c(TABLES.rpair),
          drpair = c(TABLES.drpair), maps = map(c, TABLES.maps),
          a2b = c(TABLES.a2b), wb = c(TABLES.wb), wpair = c(TABLES.wpair))
    em_i = Int32.(edge_mask)
    centers = em_i .* (senders .+ Int32(1)) .+ (Int32(1) .- em_i)
    neighbors = em_i .* (receivers .+ Int32(1)) .+ (Int32(1) .- em_i)
    site = site_energies(positions, centers, neighbors, MAX_ATOMS, tb, edge_mask)

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


# Random weights stand in for a fit; acefit! drops in here.
model = ace1_model(elements = ELEMENTS, order = ORDER, totaldegree = TOTALDEGREE)
m = model.model
ps, st = model.ps, model.st
rng = MersenneTwister(SEED)
# Scaled to ~1 eV/atom.
ps.WB .= 0.02 .* randn(rng, size(ps.WB))
ps.Wpair .= 0.02 .* randn(rng, size(ps.Wpair))
z = m._i2z[1]
RCUT = m.rbasis.rin0cuts[1, 1].rcut
println("z = $z  rcut = $RCUT  basis $(length(ps.WB)) + pair $(length(ps.Wpair))")

R_GRID, RNL, DRNL = radial_table(m.rbasis, ps.rbasis, st.rbasis, RCUT)
P_GRID, RPAIR, DRPAIR = radial_table(m.pairbasis, ps.pairbasis, st.pairbasis,
                                     m.pairbasis.rin0cuts[1, 1].rcut)
HR = (R_GRID[1], R_GRID[2] - R_GRID[1], Float64(length(R_GRID)))
HP = (P_GRID[1], P_GRID[2] - P_GRID[1], Float64(length(P_GRID)))
A_NL = [t[1] for t in m.tensor.abasis.spec]
A_LM = [t[2] for t in m.tensor.abasis.spec]
AA_SPECS = [reduce(hcat, collect.(spec))' for spec in m.tensor.aabasis.specs]
A2B = Matrix(m.tensor.A2Bmaps[1])
E0 = m.Vref.E0[z]
LMAX = isqrt(maximum(A_LM) - 1)
EXPONENTS = [[(a, b, c) for a in 0:l for b in 0:l for c in 0:l if a + b + c == l]
             for l in 0:LMAX]

# Per-l least squares from monomials onto SpheriCart probes is exact.
probes = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:64]
y_ref = Matrix(P4ML.evaluate(m.ybasis, probes))
y_scaled = Matrix(P4ML.evaluate(m.ybasis, [2.0 * v for v in probes[1:5]]))
@assert maximum(abs.(y_scaled - y_ref[1:5, :])) < 1e-12 "expected spherical Ylm"
probe_mat = permutedims(reduce(hcat, [collect(v) for v in probes]))
YLM_MAPS = map(0:LMAX) do l
    feats = monomials(probe_mat, EXPONENTS[l + 1])
    block = y_ref[:, l * l + 1:(l + 1) * (l + 1)]
    m_l = feats \ block
    @assert maximum(abs.(feats * m_l - block)) < 1e-10 "Ylm alignment failed"
    m_l
end
TABLES = (rnl = RNL, drnl = DRNL, rpair = RPAIR, drpair = DRPAIR,
          maps = Tuple(YLM_MAPS), a2b = A2B, wb = ps.WB[:, 1],
          wpair = ps.Wpair[:, 1])

a0 = 4.05
cell = [SVector(0.0, 0.0, 0.0), SVector(0.0, 0.5, 0.5),
        SVector(0.5, 0.0, 0.5), SVector(0.5, 0.5, 0.0)]
pos = [a0 * (SVector(i, j, k) + b) + 0.05 * SVector{3}(randn(rng, 3))
       for i in 0:1, j in 0:1, k in 0:1 for b in cell]
n_check = length(pos)
reference_energy = 0.0
reference_forces = zeros(3, n_check)
for i in 1:n_check
    js = [j for j in 1:n_check if j != i && norm(pos[j] - pos[i]) < RCUT]
    Rs = [pos[j] - pos[i] for j in js]
    Ei, gi = M.evaluate_ed(m, Rs, fill(z, length(Rs)), z, ps, st)
    global reference_energy += Ei
    for (g, j) in zip(gi, js)
        reference_forces[:, j] -= g
        reference_forces[:, i] += g
    end
end

positions = reduce(hcat, [collect(p) for p in pos])
edges = [(i, j) for i in 1:n_check for j in 1:n_check
         if i != j && norm(pos[i] - pos[j]) < RCUT]
site = site_energies(positions, Int32.(first.(edges)), Int32.(last.(edges)),
                     n_check, TABLES, nothing)
energy = sum(site)
println("parity vs model: E $reference_energy vs $energy " *
        "(d $(abs(reference_energy - energy)))")
@assert abs(reference_energy - energy) < 1e-12 * abs(reference_energy)

# The parity cluster padded to capacity is the example input.
args = (Reactant.to_rarray([positions zeros(3, MAX_ATOMS - n_check)]),
        Reactant.to_rarray(zeros(Int32, MAX_ATOMS)),
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
    force_error = maximum(abs.(Array(f_c)[:, 1:n_check] - reference_forces))
    println("compiled parity: E $(Float64(e_c)) max |dF| $force_error eV/A")
    @assert abs(Float64(e_c) - reference_energy) < 1e-11 * abs(reference_energy)
    # Hermite tables resample the model splines; measured floor 6e-11 at 32768 knots.
    @assert force_error < 1e-9
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
    "cutoff" => RCUT,
    "unit_style" => "metal",
    "precision" => "float64",
    "force_output" => "atom-force",
    "newton" => "on",
    "n_hops" => N_HOPS,
    "edge_pairing" => "full",
    "comm_widths" => Int[],
    "custom_call_targets" => String[],
    "uses_box" => false,
    "n_species" => 1))
