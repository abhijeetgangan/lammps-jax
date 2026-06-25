import argparse
import functools
import logging
import os
import time
from functools import partial

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = ""
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

import e3nn_jax._src.activation as _e3nn_act
import jax
import jax.numpy as jnp
from ase.build import bulk
from jax import grad, lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from nequix.calculator import NequixCalculator
from nequix.data import atomic_numbers_to_indices

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

from jax_md import custom_partition, partition, quantity, space
from jax_md._nn.util import neighbor_list_featurizer
from lammps_jax.dist.parallel import ghost_exchange
from lammps_jax.dist.parallel import replicate_data
from lammps_jax.dist.parallel import unified_ghost_exchange as uge

_e3nn_act.normalize_function = functools.lru_cache(maxsize=None)(_e3nn_act.normalize_function)


def build_system(repeat_factor, cutoff, atomic_numbers, use_custom_partition=True):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(repeat_factor)
    box = jnp.asarray(atoms.cell.T, dtype=jnp.float32)
    R = jnp.asarray(atoms.get_scaled_positions(), dtype=jnp.float32)
    z_map = atomic_numbers_to_indices(atomic_numbers)
    species = jnp.asarray([z_map[int(z)] for z in atoms.get_atomic_numbers()], dtype=jnp.int32)
    n_atoms = R.shape[0]
    idx = jnp.arange(n_atoms, dtype=R.dtype)
    R = (R + 0.01 * jnp.stack([jnp.sin(idx), jnp.cos(1.3 * idx), jnp.sin(0.7 * idx + 0.2)], 1)) % 1.0
    R = R[jnp.argsort(R[:, 0])]
    if use_custom_partition:
        disp_free, _ = space.free()
        max_nbrs = custom_partition.estimate_max_neighbors_from_box(box, cutoff, n_atoms=n_atoms, safety_factor=2.0)
        nbr_fn = custom_partition.neighbor_list_multi_image(disp_free, box, cutoff, format=partition.Sparse, fractional_coordinates=True, max_neighbors=max_nbrs)
        nbrs = nbr_fn.allocate(R)
        featurizer = custom_partition.graph_featurizer(space.free()[0])
    else:
        displacement, _ = space.periodic_general(box, fractional_coordinates=True)
        nbr_fn = partition.neighbor_list(displacement, box, cutoff, fractional_coordinates=True, format=partition.Sparse)
        nbrs = nbr_fn.allocate(R)
        while nbrs.did_buffer_overflow:
            nbrs = nbr_fn.allocate(R, extra_capacity=nbrs.idx.shape[-1] // 4)
        featurizer = neighbor_list_featurizer(displacement)
    species_pad = jnp.concatenate([species, jnp.zeros(1, dtype=species.dtype)])
    return R, box, species, species_pad, nbrs, featurizer, n_atoms


def time_force(fn, warmup, steps):
    for _ in range(warmup):
        fn().block_until_ready()
    t0 = time.perf_counter()
    value = None
    for _ in range(steps):
        value = fn().block_until_ready()
    return value, time.perf_counter() - t0


def make_energy(model, species, species_pad, nbrs, featurizer, n_atoms):
    def energy_fn(pos, **kwargs):
        graph = featurizer(species, pos, nbrs, **kwargs)
        return jnp.sum(model.node_energies(graph.edges, species_pad, graph.senders, graph.receivers)[:n_atoms])

    return energy_fn


def make_rd_force(model, species, species_pad, nbrs, featurizer, n_atoms, n_ranks, box, cutoff):
    if n_ranks == 1:
        energy_fn = make_energy(model, species, species_pad, nbrs, featurizer, n_atoms)
        return jax.jit(lambda x: -grad(energy_fn)(x))
    n_per = n_atoms // n_ranks

    def domain_energy(R_all, local_start, n_local):
        graph = featurizer(species, R_all, nbrs)
        node_energy = model.node_energies(graph.edges, species_pad, graph.senders, graph.receivers)
        return jnp.sum(lax.dynamic_slice_in_dim(node_energy[:n_atoms], local_start, n_local, axis=0))

    config = replicate_data.create_config(n_domains=n_ranks, n_atoms_per_domain=n_per, box_size=jnp.diag(box), r_cutoff=float(cutoff))
    return replicate_data.make_sharded_force(domain_energy, config, use_reduce_scatter=True)


def graph_edges(R, nbrs, use_custom_partition):
    if use_custom_partition:
        edge_mask = custom_partition.neighbor_list_multi_image_mask(nbrs)
        return nbrs.senders, nbrs.receivers, nbrs.shifts, edge_mask
    edge_mask = partition.neighbor_list_mask(nbrs)
    senders, receivers = nbrs.idx[1], nbrs.idx[0]
    safe_s = jnp.where(edge_mask, senders, 0)
    safe_r = jnp.where(edge_mask, receivers, 0)
    dr = R[safe_r] - R[safe_s]
    shifts = jnp.where(edge_mask[:, None], jnp.round(dr), 0.0)
    return senders, receivers, shifts, edge_mask


def run_rd(model, cutoff, atomic_numbers, repeats, ranks, steps, warmup):
    n_devices = jax.local_device_count()
    ranks = ranks or sorted({2 ** i for i in range(n_devices.bit_length()) if 2 ** i <= n_devices})
    summaries = []
    for rep in repeats:
        R, box, species, species_pad, nbrs, featurizer, n_atoms = build_system(rep, cutoff, atomic_numbers)
        ref_force = make_rd_force(model, species, species_pad, nbrs, featurizer, n_atoms, 1, box, cutoff)
        F_ref, t_ref = time_force(lambda: ref_force(R), warmup, steps)
        summaries.append(("rd", rep, "ref", n_atoms, 0.0, t_ref))
        for n_ranks in sorted(ranks):
            if n_ranks == 1 or n_ranks > n_devices or n_atoms % n_ranks != 0:
                continue
            force_fn = make_rd_force(model, species, species_pad, nbrs, featurizer, n_atoms, n_ranks, box, cutoff)
            F_par, elapsed = time_force(lambda: force_fn(R), warmup, steps)
            f_err = float(jnp.max(jnp.abs(F_par - F_ref)))
            summaries.append(("rd", rep, n_ranks, n_atoms, f_err, elapsed))
    return summaries


def run_unified(model, cutoff, atomic_numbers, repeats, ranks, steps, warmup, use_custom_partition=True):
    n_devices = jax.local_device_count()
    ranks = ranks or sorted({2 ** i for i in range(n_devices.bit_length()) if 2 ** i <= n_devices})
    summaries = []
    for rep in repeats:
        R, box, species, species_pad, nbrs, featurizer, n_atoms = build_system(rep, cutoff, atomic_numbers, use_custom_partition)
        ref_energy = make_energy(model, species, species_pad, nbrs, featurizer, n_atoms)
        ref_force_fn = jax.jit(lambda x: -grad(ref_energy)(x))
        F_ref, t_ref = time_force(lambda: ref_force_fn(R), warmup, steps)
        summaries.append(("unified", rep, "ref", n_atoms, 0.0, t_ref))
        senders, receivers, shifts, edge_mask = graph_edges(R, nbrs, use_custom_partition)
        for n_ranks in sorted(ranks):
            if n_ranks == 1 or n_ranks > n_devices or n_atoms % n_ranks != 0:
                continue
            subgraph = uge.precompute_subgraphs(senders, receivers, shifts, edge_mask, n_atoms, n_ranks, 1, species)
            subgraph_arrays = subgraph[:6]
            config = uge.create_config(n_domains=n_ranks, n_atoms=n_atoms)
            force_fn = uge.make_sharded_force(model, box, config)
            F_par, elapsed = time_force(lambda: force_fn(R, *subgraph_arrays), warmup, steps)
            f_err = float(jnp.max(jnp.abs(F_par - F_ref)))
            summaries.append(("unified", rep, n_ranks, n_atoms, f_err, elapsed))
    return summaries


def run_demo(model, cutoff, atomic_numbers, steps, warmup):
    R, box, species, species_pad, nbrs, featurizer, n_atoms = build_system(2, cutoff, atomic_numbers)
    n_ranks = max(d for d in range(2, jax.local_device_count() + 1) if n_atoms % d == 0)
    force_single = make_rd_force(model, species, species_pad, nbrs, featurizer, n_atoms, 1, box, cutoff)
    F_ref, t_ref = time_force(lambda: force_single(R), warmup, steps)
    force_parallel = make_rd_force(model, species, species_pad, nbrs, featurizer, n_atoms, n_ranks, box, cutoff)
    F_par, t_par = time_force(lambda: force_parallel(R), warmup, steps)
    f_err = float(jnp.max(jnp.abs(F_par - F_ref)))
    stress_ref = quantity.stress(lambda pos, **kw: make_energy(model, species, species_pad, nbrs, featurizer, n_atoms)(pos, **kw), R, box)

    def sharded_energy_with_kwargs(pos, **kwargs):
        n_per = n_atoms // n_ranks

        def domain_energy(R_all, local_start, n_local):
            graph = featurizer(species, R_all, nbrs, **kwargs)
            node_energy = model.node_energies(graph.edges, species_pad, graph.senders, graph.receivers)
            return jnp.sum(lax.dynamic_slice_in_dim(node_energy[:n_atoms], local_start, n_local, axis=0))

        config = replicate_data.create_config(n_domains=n_ranks, n_atoms_per_domain=n_per, box_size=jnp.diag(box), r_cutoff=float(cutoff))
        return replicate_data.make_sharded_energy(domain_energy, config)(pos)

    stress_par = quantity.stress(sharded_energy_with_kwargs, R, box)
    stress_err = float(jnp.max(jnp.abs(stress_par - stress_ref)))
    return [("demo", 2, "ref", n_atoms, 0.0, t_ref), ("demo", 2, n_ranks, n_atoms, max(f_err, stress_err), t_par)]


def run_ghost(model, cutoff, atomic_numbers, repeat, ranks, steps, warmup):
    n_devices = jax.local_device_count()
    n_ranks = ranks[0] if ranks else n_devices
    R, box, species, species_pad, nbrs, featurizer, n_atoms = build_system(repeat, cutoff, atomic_numbers)
    if n_atoms % n_ranks != 0:
        raise ValueError("atom count must divide ranks")
    ref_energy = make_energy(model, species, species_pad, nbrs, featurizer, n_atoms)
    ref_force = jax.jit(lambda x: -grad(ref_energy)(x))
    F_ref, t_ref = time_force(lambda: ref_force(R), warmup, steps)
    edge_mask = custom_partition.neighbor_list_multi_image_mask(nbrs)
    subgraph = uge.precompute_subgraphs(nbrs.senders, nbrs.receivers, nbrs.shifts, edge_mask, n_atoms, n_ranks, 1, species)
    subgraph_arrays = subgraph[:6]
    config_uge = uge.create_config(n_domains=n_ranks, n_atoms=n_atoms)
    host_force_fn = uge.make_sharded_force(model, box, config_uge)
    F_host, t_host = time_force(lambda: host_force_fn(R, *subgraph_arrays), warmup, steps)
    n_per = n_atoms // n_ranks
    config = ghost_exchange.create_config(n_ranks, n_atoms, box, cutoff, capacity_mult=1.5)
    max_owned = config.max_owned
    max_ghost = config.max_ghost
    max_local = max_owned + max_ghost
    cells_per_side, cell_capacity, max_neighbors = ghost_exchange.estimate_nl_params(box, cutoff, max_local)
    cell_capacity = cell_capacity * n_ranks
    max_nodes = n_per + config.max_ghost + 1
    mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))
    sharding = NamedSharding(mesh, P("i"))
    exchange_fn = uge._make_exchange(n_atoms, n_per, n_ranks, "i")
    sort_order = jnp.argsort(R[:, 0])
    R_sorted = R[sort_order]
    species_sorted = species[sort_order]
    R_global_mesh = jax.device_put(R_sorted.reshape(n_ranks, n_per, 3), sharding)
    sp_global_mesh = jax.device_put(species_sorted.reshape(n_ranks, n_per), sharding)

    @partial(shard_map, mesh=mesh, in_specs=(P("i"), P("i")), out_specs=(P(), P("i")), check_vma=False)
    def ghost_exchange_ef(R_local, sp_local):
        owned_pos = jnp.squeeze(R_local, axis=0)
        owned_sp = jnp.squeeze(sp_local, axis=0)
        device_idx = lax.axis_index("i")
        atom_start = lax.convert_element_type(device_idx * n_per, jnp.int32)
        owned_gidx = atom_start + jnp.arange(n_per, dtype=jnp.int32)
        owned_data = jnp.concatenate([owned_pos, owned_gidx[:, None].astype(owned_pos.dtype)], axis=1)
        ghost_data, n_ghost = ghost_exchange.ghost_exchange(owned_data, n_per, config, "i")
        ghost_pos = ghost_data[:, :3]
        ghost_gidx = ghost_data[:, 3].astype(jnp.int32)
        ghost_sp = jnp.zeros(max_ghost, dtype=owned_sp.dtype)
        nidx, ls, lr, lsh, lsp, lem, overflow = ghost_exchange.ghost_exchange_subgraph(owned_pos, ghost_pos, owned_sp, ghost_sp, owned_gidx, ghost_gidx, n_per, n_ghost, box, cutoff, cells_per_side, cell_capacity, max_neighbors, max_local, max_nodes, 1)
        sp_pad = jnp.concatenate([lsp, jnp.zeros(1, dtype=lsp.dtype)])
        R_all = lax.all_gather(R_local, axis_name="i").reshape(-1, 3)
        safe_lr = jnp.where(lr < max_nodes, lr, 0)
        safe_ls = jnp.where(ls < max_nodes, ls, 0)
        global_r = nidx[safe_lr]
        global_s = nidx[safe_ls]
        safe_gr = jnp.where(global_r < n_atoms, global_r, 0)
        safe_gs = jnp.where(global_s < n_atoms, global_s, 0)
        lsh = jnp.round(R_all[safe_gr] - R_all[safe_gs])
        E_local, edge_grads = uge._energy_and_edge_grads(model, box, nidx, ls, lr, lsh, sp_pad, lem, n_per, n_atoms, atom_start, exchange_fn, R_all)
        E_total = lax.psum(E_local, axis_name="i")
        forces = jnp.zeros((n_atoms, 3), dtype=R_all.dtype)
        forces = forces.at[nidx[lr]].add(-edge_grads)
        forces = forces.at[nidx[ls]].add(edge_grads)
        F_local = lax.psum_scatter(forces, axis_name="i", scatter_dimension=0, tiled=True)
        return E_total, F_local

    @jax.jit
    def compute(R_m, sp_m):
        E, F = ghost_exchange_ef(R_m, sp_m)
        return E, F.reshape(-1, 3)

    F_gj, t_gj = time_force(lambda: compute(R_global_mesh, sp_global_mesh)[1], warmup, steps)
    F_gj = F_gj[jnp.argsort(sort_order)]
    f_err = float(jnp.max(jnp.abs(F_gj - F_ref)))
    host_err = float(jnp.max(jnp.abs(F_host - F_ref)))
    return [("ghost", repeat, "ref", n_atoms, 0.0, t_ref), ("ghost-host", repeat, n_ranks, n_atoms, host_err, t_host), ("ghost-jax", repeat, n_ranks, n_atoms, f_err, t_gj)]


def run(mode="unified", repeats=None, ranks=None, steps=10, warmup=3, model_name="nequix-mp-1", use_custom_partition=True):
    calc = NequixCalculator(model_name, use_kernel=(mode == "ghost"))
    model = calc.model
    cutoff = calc.cutoff
    atomic_numbers = calc.config["atomic_numbers"]
    if mode == "demo":
        return run_demo(model, cutoff, atomic_numbers, steps, warmup)
    if mode == "rd":
        return run_rd(model, cutoff, atomic_numbers, repeats or [5, 6, 7], ranks, steps, warmup)
    if mode == "unified":
        return run_unified(model, cutoff, atomic_numbers, repeats or [4, 5, 6], ranks, steps, warmup, use_custom_partition)
    if mode == "ghost":
        return run_ghost(model, cutoff, atomic_numbers, (repeats or [4])[0], ranks, steps, warmup)
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("demo", "rd", "unified", "ghost"), default="unified")
    parser.add_argument("--repeats", type=int, nargs="+", default=None)
    parser.add_argument("--ranks", type=int, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--model", default="nequix-mp-1")
    parser.add_argument("--standard-neighbor-list", action="store_true")
    args = parser.parse_args()
    for row in run(args.mode, args.repeats, args.ranks, args.steps, args.warmup, args.model, not args.standard_neighbor_list):
        mode, repeat, rank, n_atoms, f_err, elapsed = row
        print(f"{mode} repeat={repeat} rank={rank} atoms={n_atoms} f_err={f_err:.3e} time={elapsed:.3f}s")


if __name__ == "__main__":
    main()
