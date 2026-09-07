# /// script
# requires-python = ">=3.11"
# dependencies = ["lammps-jax[dist]"]
#
# [tool.uv.sources]
# lammps-jax = { path = "../../../..", editable = true }
# ///
import argparse
import functools
import logging
import os
import time
from functools import partial

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = ""
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

import e3nn_jax._src.activation as e3nn_activation
import jax
import jax.numpy as jnp
from ase.build import bulk
from jax import grad, lax, shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from nequix.calculator import NequixCalculator
from nequix.data import atomic_numbers_to_indices
from jax_md import custom_partition, partition, quantity, space
from jax_md._nn.util import neighbor_list_featurizer
from lammps_jax.dist.parallel import feature_exchange
from lammps_jax.dist.parallel import ghost_exchange
from lammps_jax.dist.parallel import replicate_data

e3nn_activation.normalize_function = functools.lru_cache(maxsize=None)(e3nn_activation.normalize_function)


def build_system(repeat_factor, cutoff, atomic_numbers, use_custom_partition=True):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(repeat_factor)
    box = jnp.asarray(atoms.cell.T, dtype=jnp.float32)
    R = jnp.asarray(atoms.get_scaled_positions(), dtype=jnp.float32)
    z_map = atomic_numbers_to_indices(atomic_numbers)
    species = jnp.asarray([z_map[int(z)] for z in atoms.get_atomic_numbers()], dtype=jnp.int32)
    n_atoms = R.shape[0]
    idx = jnp.arange(n_atoms, dtype=R.dtype)
    half_plane = jnp.array([1.0 / (8 * repeat_factor), 0.0, 0.0], dtype=R.dtype)
    R = (R + half_plane + 0.01 * jnp.stack([jnp.sin(idx), jnp.cos(1.3 * idx), jnp.sin(0.7 * idx + 0.2)], 1)) % 1.0
    R = R[jnp.argsort(R[:, 0])]
    if use_custom_partition:
        disp_free = space.free()[0]
        max_neighbors = custom_partition.estimate_max_neighbors_from_box(box, cutoff, n_atoms=n_atoms, safety_factor=2.0)
        nbr_fn = custom_partition.neighbor_list_multi_image(disp_free, box, cutoff, format=partition.Sparse, fractional_coordinates=True, max_neighbors=max_neighbors)
        nbrs = nbr_fn.allocate(R)
        featurizer = custom_partition.graph_featurizer(space.free()[0])
    else:
        displacement = space.periodic_general(box, fractional_coordinates=True)[0]
        nbr_fn = partition.neighbor_list(displacement, box, cutoff, fractional_coordinates=True, format=partition.Sparse)
        nbrs = nbr_fn.allocate(R)
        while nbrs.did_buffer_overflow:
            nbrs = nbr_fn.allocate(R, extra_capacity=nbrs.idx.shape[-1] // 4)
        featurizer = neighbor_list_featurizer(displacement)
    species_pad = jnp.concatenate([species, jnp.zeros(1, dtype=species.dtype)])
    return R, box, species, species_pad, nbrs, featurizer, n_atoms


def time_force(fn, warmup, steps):
    for step in range(warmup):
        fn().block_until_ready()
    t0 = time.perf_counter()
    value = None
    for step in range(steps):
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

    config = replicate_data.create_config(n_domains=n_ranks, n_atoms_per_domain=n_per)
    return replicate_data.make_sharded_force(domain_energy, config, use_reduce_scatter=True)


def graph_edges(R, nbrs, use_custom_partition):
    if use_custom_partition:
        edge_mask = custom_partition.neighbor_list_multi_image_mask(nbrs)
        return nbrs.senders, nbrs.receivers, nbrs.shifts, edge_mask
    edge_mask = partition.neighbor_list_mask(nbrs)
    senders, receivers = nbrs.idx[1], nbrs.idx[0]
    safe_senders = jnp.where(edge_mask, senders, 0)
    safe_receivers = jnp.where(edge_mask, receivers, 0)
    dr = R[safe_receivers] - R[safe_senders]
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
            n_per = n_atoms // n_ranks
            subgraph = feature_exchange.precompute_subgraphs(senders, receivers, shifts, edge_mask, n_atoms, n_ranks, 1, species)
            subgraph_arrays = subgraph[:6]
            config = replicate_data.create_config(n_domains=n_ranks, n_atoms_per_domain=n_per)
            force_fn = feature_exchange.make_sharded_force(model, box, config)
            placed = place_subgraph(R, subgraph_arrays, config)
            F_par, elapsed = time_force(lambda: force_fn(*placed), warmup, steps)
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

        config = replicate_data.create_config(n_domains=n_ranks, n_atoms_per_domain=n_per)
        return replicate_data.make_sharded_energy(domain_energy, config)(pos)

    stress_par = quantity.stress(sharded_energy_with_kwargs, R, box)
    stress_err = float(jnp.max(jnp.abs(stress_par - stress_ref)))
    return [("demo", 2, "ref", n_atoms, 0.0, t_ref), ("demo", 2, n_ranks, n_atoms, max(f_err, stress_err), t_par)]


def place_subgraph(R, subgraph_arrays, config):
    """Shards positions and the static subgraph arrays once so the timed calls move nothing."""
    sharding = NamedSharding(config.mesh, P(config.axis_name))
    R_mesh = R.reshape(config.n_domains, config.n_atoms_per_domain, 3)
    return [jax.device_put(array, sharding) for array in (R_mesh, *subgraph_arrays)]


def run_ghost(model, cutoff, atomic_numbers, repeat, ranks, steps, warmup, skin=0.5):
    n_devices = jax.local_device_count()
    n_ranks = ranks[0] if ranks else n_devices
    R, box, species, species_pad, nbrs, featurizer, n_atoms = build_system(repeat, cutoff, atomic_numbers)
    if n_atoms % n_ranks != 0:
        raise ValueError("n_ranks must divide the atom count")
    n_per = n_atoms // n_ranks
    ref_energy = make_energy(model, species, species_pad, nbrs, featurizer, n_atoms)
    ref_force = jax.jit(lambda x: -grad(ref_energy)(x))
    F_ref, t_ref = time_force(lambda: ref_force(R), warmup, steps)
    edge_mask = custom_partition.neighbor_list_multi_image_mask(nbrs)
    subgraph = feature_exchange.precompute_subgraphs(nbrs.senders, nbrs.receivers, nbrs.shifts, edge_mask, n_atoms, n_ranks, 1, species)
    feature_config = replicate_data.create_config(n_domains=n_ranks, n_atoms_per_domain=n_per)
    host_force_fn = feature_exchange.make_sharded_force(model, box, feature_config)
    placed = place_subgraph(R, subgraph[:6], feature_config)
    F_host, t_host = time_force(lambda: host_force_fn(*placed), warmup, steps)
    config = ghost_exchange.create_config(n_ranks, n_atoms, box, cutoff, capacity_mult=1.5, skin=skin)
    cells_per_side, cell_capacity, max_neighbors, max_edges = ghost_exchange.subgraph_params(config, box, n_atoms, capacity_mult=2.0)
    mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))
    sharding = NamedSharding(mesh, P("i"))
    ids = jnp.arange(n_atoms, dtype=jnp.int32)
    owned, owned_ints, counts, rank_of = ghost_exchange.pack_tiles(R, config, ints=jnp.stack([species, ids], 1))
    owned_mesh, ints_mesh, counts_mesh = (jax.device_put(jnp.asarray(array), sharding) for array in (owned, owned_ints, counts))
    node_energy_fn = partial(feature_exchange.node_energies, model)
    rows = config.max_owned

    @partial(shard_map, mesh=mesh, in_specs=(P("i"), P("i"), P("i")), out_specs=(P("i"),) * 7 + (P(),), check_vma=False)
    def rebuild(owned_stack, ints_stack, counts_arr):
        """Migrates, re-records the plan and rebuilds the edge list: the full neighbour-list rebuild."""
        owned_pos, owned_ints_local, n_owned, migrate_flags = ghost_exchange.redistribute(
            owned_stack.squeeze(0), counts_arr[0], config, "i", owned_ints=ints_stack.squeeze(0))
        ghost_data, n_ghost, exchange_flags, plan = ghost_exchange.ghost_exchange(owned_pos, n_owned, config, "i")
        ints_nodes = ghost_exchange.exchange_apply(plan, owned_ints_local, config, "i")
        ghost_ints = lax.dynamic_slice_in_dim(ints_nodes, n_owned, config.max_ghost, axis=0)
        node_idx, senders, receivers, node_species, edge_mask, overflow = ghost_exchange.ghost_exchange_subgraph(
            owned_pos, ghost_data, owned_ints_local[:, 0], ghost_ints[:, 0], owned_ints_local[:, 1], ghost_ints[:, 1],
            n_owned, n_ghost, box, cells_per_side, cell_capacity, max_neighbors, max_edges, config, "i")
        flags = lax.psum(jnp.stack([migrate_flags[0] | exchange_flags[0], jnp.any(migrate_flags[1:]) | exchange_flags[1] | overflow]).astype(jnp.int32), "i")
        stacked = jax.tree.map(lambda array: array[None], (owned_pos, owned_ints_local, n_owned, plan, senders, receivers, edge_mask))
        return (*stacked, flags)

    @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 7, out_specs=(P(), P("i")), check_vma=False)
    def step(owned_stack, ints_stack, counts_arr, plan, senders, receivers, edge_mask):
        owned_pos, owned_species, n_owned = (owned_stack.squeeze(0), ints_stack.squeeze(0)[:, 0], counts_arr[0])
        plan, senders, receivers, edge_mask = jax.tree.map(lambda array: array.squeeze(0), (plan, senders, receivers, edge_mask))
        E_local, grad_owned = jax.value_and_grad(lambda pos: feature_exchange.ghost_energy(
            node_energy_fn, pos, n_owned, owned_species, senders, receivers, edge_mask, plan, box, config, "i"))(owned_pos)
        return lax.psum(E_local, "i"), -grad_owned[None]

    rebuild = jax.jit(rebuild)
    step = jax.jit(step)
    *state, flags = rebuild(owned_mesh, ints_mesh, counts_mesh)
    if int(flags[0]):
        raise RuntimeError("an owned atom lies outside its tile after redistribute")
    if int(flags[1]):
        raise RuntimeError("ghost exchange overflow; raise capacity_mult, cell_capacity, max_neighbors or max_edges")
    _, t_rebuild = time_force(lambda: rebuild(owned_mesh, ints_mesh, counts_mesh)[-1], warmup, steps)
    F_owned, t_ghost = time_force(lambda: step(*state)[1], warmup, steps)
    owned_ints, counts = (jnp.asarray(state[1]), jnp.asarray(state[2]))
    F_ghost = jnp.zeros((n_atoms, 3), dtype=F_owned.dtype)
    for domain in range(n_ranks):
        live = int(counts[domain])
        F_ghost = F_ghost.at[owned_ints[domain, :live, 1]].set(F_owned[domain, :live])
    f_err = float(jnp.max(jnp.abs(F_ghost - F_ref)))
    host_err = float(jnp.max(jnp.abs(F_host - F_ref)))
    return [("ghost", repeat, "ref", n_atoms, 0.0, t_ref), ("ghost-host", repeat, n_ranks, n_atoms, host_err, t_host),
            ("ghost-rebuild", repeat, n_ranks, n_atoms, 0.0, t_rebuild), ("ghost-step", repeat, n_ranks, n_atoms, f_err, t_ghost)]


def run(mode="unified", repeats=None, ranks=None, steps=10, warmup=3, model_name="nequix-mp-1", use_custom_partition=True, skin=0.5):
    calc = NequixCalculator(model_name, use_kernel=False)
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
        return run_ghost(model, cutoff, atomic_numbers, (repeats or [4])[0], ranks, steps, warmup, skin)
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
    parser.add_argument("--skin", type=float, default=0.5, help="Verlet skin in Angstrom for --mode ghost")
    args = parser.parse_args()
    for row in run(args.mode, args.repeats, args.ranks, args.steps, args.warmup, args.model, not args.standard_neighbor_list, args.skin):
        mode, repeat, rank, n_atoms, f_err, elapsed = row
        print(f"{mode} repeat={repeat} rank={rank} atoms={n_atoms} f_err={f_err:.3e} time={elapsed:.3f}s")


if __name__ == "__main__":
    main()
