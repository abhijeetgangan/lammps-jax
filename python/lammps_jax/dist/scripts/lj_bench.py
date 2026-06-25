import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax_md import energy, partition, space

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

from lammps_jax.dist.parallel import replicate_data
from lammps_jax.dist.parallel import unified_ghost_exchange as uge

R_CUTOFF = 2.5
R_ONSET = 2.0


def build_fcc(n_cells, a=1.5):
    basis = jnp.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    shifts = jnp.array([[i, j, k] for i in range(n_cells) for j in range(n_cells) for k in range(n_cells)])
    R_frac = (shifts[:, None, :] + basis[None, :, :]).reshape(-1, 3) / n_cells
    box = jnp.eye(3) * (n_cells * a)
    return R_frac[jnp.argsort(R_frac[:, 0])], box


def build_neighbor_idx(displacement_fn, R_frac, box, cutoff):
    nbr_fn = partition.neighbor_list(displacement_fn, box, cutoff, fractional_coordinates=True, format=partition.Sparse)
    nbrs = nbr_fn.allocate(R_frac)
    mask = partition.neighbor_list_mask(nbrs)
    receivers, senders = nbrs.idx[0][mask], nbrs.idx[1][mask]
    return jnp.stack([senders, receivers])


def time_force(fn, warmup, steps):
    for _ in range(warmup):
        fn().block_until_ready()
    t0 = time.perf_counter()
    value = None
    for _ in range(steps):
        value = fn().block_until_ready()
    return value, time.perf_counter() - t0


def make_rd_energy_force(domain_energy, n_ranks, n_per):
    mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))
    sharding = NamedSharding(mesh, P("i"))

    @partial(shard_map, mesh=mesh, in_specs=(P("i"), P("i")), out_specs=(P(), P("i")), check_vma=False)
    def _ef(R_local, nbrs_local):
        device_idx = lax.axis_index("i")
        R_all = lax.all_gather(R_local, axis_name="i").reshape(-1, 3)
        nbrs_local = jnp.squeeze(nbrs_local, axis=0)
        start = lax.convert_element_type(device_idx * n_per, jnp.int32)
        E_local, grad_all = jax.value_and_grad(lambda r: domain_energy(r, nbrs_local, start, n_per))(R_all)
        E_total = lax.psum(E_local, axis_name="i")
        F_local = -lax.psum_scatter(grad_all, axis_name="i", scatter_dimension=0, tiled=True)
        return E_total, F_local

    @jax.jit
    def compute(R_mesh, nbrs_mesh):
        E, F = _ef(R_mesh, nbrs_mesh)
        return E, F.reshape(-1, 3)

    return compute, sharding


def run_rd(cells, ranks, steps, warmup):
    n_devices = jax.local_device_count()
    ranks = ranks or sorted({2 ** i for i in range(n_devices.bit_length()) if 2 ** i <= n_devices})
    pair_fn = energy.multiplicative_isotropic_cutoff(energy.lennard_jones, r_onset=R_ONSET, r_cutoff=R_CUTOFF)
    summaries = []
    for nc in cells:
        jax.clear_caches()
        R, box = build_fcc(nc)
        n_atoms = R.shape[0]
        displacement, _ = space.periodic_general(box, fractional_coordinates=True)
        domain_energy = replicate_data.domain_energy_from_pair_with_neighbors(pair_fn, displacement)
        neighbor_idx = build_neighbor_idx(displacement, R, box, R_CUTOFF)
        ref_energy_fn = jax.jit(lambda x: domain_energy(x, neighbor_idx, 0, n_atoms))
        ref_force_fn = jax.jit(lambda x: -grad(lambda r: domain_energy(r, neighbor_idx, 0, n_atoms))(x))
        E_ref = ref_energy_fn(R).block_until_ready()
        F_ref, t_ref = time_force(lambda: ref_force_fn(R), warmup, steps)
        summaries.append(("rd", nc, "ref", n_atoms, float(E_ref), 0.0, 0.0, t_ref))
        for n_ranks in sorted(ranks):
            if n_ranks == 1 or n_ranks > n_devices or n_atoms % n_ranks != 0:
                continue
            n_per = n_atoms // n_ranks
            sharded_nbrs = replicate_data.shard_neighbor_idx_by_sender(neighbor_idx, n_domains=n_ranks, n_per_domain=n_per)
            par_ef, sharding = make_rd_energy_force(domain_energy, n_ranks, n_per)
            R_mesh = jax.device_put(R.reshape(n_ranks, n_per, 3), sharding)
            nbrs_mesh = jax.device_put(sharded_nbrs, sharding)
            E_par, F_par = par_ef(R_mesh, nbrs_mesh)
            E_par.block_until_ready()
            F_par = F_par.block_until_ready()
            _, elapsed = time_force(lambda: par_ef(R_mesh, nbrs_mesh)[1], warmup, steps)
            e_err = float(jnp.abs(E_par - E_ref))
            f_err = float(jnp.max(jnp.abs(F_par - F_ref)))
            summaries.append(("rd", nc, n_ranks, n_atoms, float(E_par), e_err, f_err, elapsed))
    return summaries


def make_eam_domain_energy(displacement):
    metric = space.metric(displacement)
    charge = lambda dr: jnp.where(dr < R_CUTOFF, jnp.exp(-dr), 0.0)
    embed = lambda rho: -jnp.sqrt(rho + 1e-6)
    pair = lambda dr: jnp.where(dr < R_CUTOFF, jnp.exp(-2.0 * dr), 0.0)

    def domain_energy(R_all, local_start, n_local):
        n_all = R_all.shape[0]
        start = lax.convert_element_type(local_start, jnp.int32)
        R_local = lax.dynamic_slice_in_dim(R_all, start, n_local, axis=0)
        dr = jax.vmap(lambda r_i: jax.vmap(lambda r_j: metric(r_i, r_j))(R_all))(R_local)
        rho = jnp.sum(charge(dr), axis=1)
        E_embed = jnp.sum(embed(rho))
        idx_g = jnp.arange(n_all, dtype=jnp.int32)
        idx_l = start + jnp.arange(n_local, dtype=jnp.int32)
        diag = idx_l[:, None] == idx_g[None, :]
        safe_dr = jnp.where(diag, jnp.ones_like(dr), dr)
        E_pair = 0.5 * jnp.sum(jnp.where(diag, 0.0, pair(safe_dr)))
        return E_embed + E_pair

    return domain_energy


def run_eam(cells, ranks, steps, warmup):
    n_devices = jax.local_device_count()
    ranks = ranks or sorted({2 ** i for i in range(n_devices.bit_length()) if 2 ** i <= n_devices})
    summaries = []
    for nc in cells:
        R, box = build_fcc(nc)
        n_atoms = R.shape[0]
        displacement, _ = space.periodic_general(box, fractional_coordinates=False)
        domain_energy = make_eam_domain_energy(displacement)
        E_ref = domain_energy(R, 0, n_atoms).block_until_ready()
        F_ref_fn = jax.jit(lambda x: -grad(lambda r: domain_energy(r, 0, n_atoms))(x))
        F_ref, t_ref = time_force(lambda: F_ref_fn(R), warmup, steps)
        summaries.append(("eam", nc, "ref", n_atoms, float(E_ref), 0.0, 0.0, t_ref))
        for n_ranks in sorted(ranks):
            if n_ranks == 1 or n_ranks > n_devices or n_atoms % n_ranks != 0:
                continue
            n_per = n_atoms // n_ranks
            config = replicate_data.create_config(n_domains=n_ranks, n_atoms_per_domain=n_per, box_size=jnp.diag(box), r_cutoff=R_CUTOFF)
            par_ef = replicate_data.make_sharded_energy_force(domain_energy, config, use_reduce_scatter=True)
            E_par, F_par = par_ef(R)
            E_par.block_until_ready()
            F_par = F_par.block_until_ready()
            _, elapsed = time_force(lambda: par_ef(R)[1], warmup, steps)
            e_err = float(jnp.abs(E_par - E_ref))
            f_err = float(jnp.max(jnp.abs(F_par - F_ref)))
            summaries.append(("eam", nc, n_ranks, n_atoms, float(E_par), e_err, f_err, elapsed))
    return summaries


def build_unified_system(repeat_factor):
    from ase.build import bulk

    atoms = bulk("Ar", "fcc", a=1.5, cubic=True).repeat(repeat_factor)
    box = jnp.asarray(atoms.cell.T, dtype=jnp.float32)
    R = jnp.asarray(atoms.get_scaled_positions(), dtype=jnp.float32)
    n_atoms = R.shape[0]
    idx = jnp.arange(n_atoms, dtype=R.dtype)
    pert = 0.01 / repeat_factor
    R = (R + pert * jnp.stack([jnp.sin(idx), jnp.cos(1.3 * idx), jnp.sin(0.7 * idx + 0.2)], 1)) % 1.0
    R = R[jnp.argsort(R[:, 0])]
    displacement, _ = space.periodic_general(box, fractional_coordinates=True)
    nbr_fn = partition.neighbor_list(displacement, box, R_CUTOFF, fractional_coordinates=True, format=partition.Sparse)
    nbrs = nbr_fn.allocate(R)
    while nbrs.did_buffer_overflow:
        nbrs = nbr_fn.allocate(R, extra_capacity=nbrs.idx.shape[-1] // 4)
    return R, box, nbrs, n_atoms


def run_unified(repeats, ranks, steps, warmup):
    n_devices = jax.local_device_count()
    ranks = ranks or sorted({2 ** i for i in range(n_devices.bit_length()) if 2 ** i <= n_devices})
    pair_fn = energy.multiplicative_isotropic_cutoff(energy.lennard_jones, r_onset=R_ONSET, r_cutoff=R_CUTOFF)
    summaries = []
    for rep in repeats:
        R, box, nbrs, n_atoms = build_unified_system(rep)
        displacement, _ = space.periodic_general(box, fractional_coordinates=True)
        metric = space.metric(displacement)
        mask = partition.neighbor_list_mask(nbrs)
        senders, receivers = nbrs.idx[1], nbrs.idx[0]

        def ref_energy(pos):
            dr = jax.vmap(metric)(pos[senders], pos[receivers])
            safe_dr = jnp.where(mask, dr, jnp.ones_like(dr))
            return 0.5 * jnp.sum(jnp.where(mask, pair_fn(safe_dr), 0.0))

        ref_force_fn = jax.jit(lambda x: -grad(ref_energy)(x))
        E_ref = float(jax.jit(ref_energy)(R).block_until_ready())
        F_ref, t_ref = time_force(lambda: ref_force_fn(R), warmup, steps)
        summaries.append(("unified", rep, "ref", n_atoms, E_ref, 0.0, 0.0, t_ref))
        for n_ranks in sorted(ranks):
            if n_ranks == 1 or n_ranks > n_devices or n_atoms % n_ranks != 0:
                continue
            n_per = n_atoms // n_ranks
            shifts = jnp.zeros((nbrs.idx.shape[1], 3), dtype=R.dtype)
            sg = uge.precompute_subgraphs(senders, receivers, shifts, mask, n_atoms, n_ranks, 1, jnp.zeros(n_atoms, dtype=jnp.int32))
            dom_nidx, dom_s, dom_r, dom_sh, dom_sp, dom_em = sg[:6]
            config = uge.create_config(n_domains=n_ranks, n_atoms=n_atoms)
            mesh = config.mesh
            axis_name = config.axis_name
            sharding = NamedSharding(mesh, P(axis_name))

            @partial(shard_map, mesh=mesh, in_specs=(P(axis_name),) + (P(axis_name),) * 6, out_specs=(P(), P(axis_name)), check_vma=False)
            def _sharded_ef(R_local, nidx, ls, lr, lsh, lsp, lem):
                R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
                nidx = nidx.squeeze(0)
                ls = ls.squeeze(0)
                lr = lr.squeeze(0)
                lem = lem.squeeze(0)
                pos_frac = R_all[nidx]

                def lj_energy(edge_vectors):
                    dist_sq = jnp.sum(edge_vectors ** 2, axis=-1)
                    dist = jnp.sqrt(jnp.where(dist_sq > 0, dist_sq, 1.0))
                    valid = lem & (dist_sq > 0)
                    e = jnp.where(valid, pair_fn(dist), 0.0)
                    owned = (lr >= 0) & (lr < n_per)
                    return 0.5 * jnp.sum(e * owned.astype(e.dtype))

                dr_frac = pos_frac[lr] - pos_frac[ls]
                dr_frac = dr_frac - jnp.round(dr_frac)
                dR = dr_frac @ box
                E_local, edge_grads = jax.value_and_grad(lj_energy)(dR)
                E_total = lax.psum(E_local, axis_name=axis_name)
                forces = jnp.zeros((n_atoms, 3), dtype=R_all.dtype)
                forces = forces.at[nidx[lr]].add(-edge_grads)
                forces = forces.at[nidx[ls]].add(edge_grads)
                F_local = lax.psum_scatter(forces, axis_name=axis_name, scatter_dimension=0, tiled=True)
                return E_total, F_local

            @jax.jit
            def force_fn(R_flat, nidx, ls, lr, lsh, lsp, lem):
                return _sharded_ef(
                    jax.device_put(R_flat.reshape(n_ranks, n_per, 3), sharding),
                    jax.device_put(nidx, sharding),
                    jax.device_put(ls, sharding),
                    jax.device_put(lr, sharding),
                    jax.device_put(lsh, sharding),
                    jax.device_put(lsp, sharding),
                    jax.device_put(lem, sharding),
                )

            E_par, F_par = force_fn(R, dom_nidx, dom_s, dom_r, dom_sh, dom_sp, dom_em)
            E_par = float(E_par.block_until_ready())
            F_par = F_par.reshape(-1, 3).block_until_ready()
            _, elapsed = time_force(lambda: force_fn(R, dom_nidx, dom_s, dom_r, dom_sh, dom_sp, dom_em)[1], warmup, steps)
            e_err = abs(E_par - E_ref)
            f_err = float(jnp.max(jnp.abs(F_par - F_ref)))
            summaries.append(("unified", rep, n_ranks, n_atoms, E_par, e_err, f_err, elapsed))
    return summaries


def run(mode="rd", sizes=None, ranks=None, steps=10, warmup=3):
    if mode == "rd":
        return run_rd(sizes or [4, 5, 6], ranks, steps, warmup)
    if mode == "unified":
        return run_unified(sizes or [4, 5, 6], ranks, steps, warmup)
    if mode == "eam":
        return run_eam(sizes or [2, 3, 4], ranks, steps, warmup)
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rd", "unified", "eam"), default="rd")
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--ranks", type=int, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    for row in run(args.mode, args.sizes, args.ranks, args.steps, args.warmup):
        mode, size, rank, n_atoms, energy_value, e_err, f_err, elapsed = row
        print(f"{mode} size={size} rank={rank} atoms={n_atoms} energy={energy_value:.6e} e_err={e_err:.3e} f_err={f_err:.3e} time={elapsed:.3f}s")


if __name__ == "__main__":
    main()
