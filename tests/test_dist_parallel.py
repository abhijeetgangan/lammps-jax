"""Multi-device validation of dist.parallel on 8 fake CPU devices in float64.

pytest re-runs this file in a subprocess; skips without the [dist] extra.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax_md")

PYTHON_DIR = str(Path(__file__).resolve().parent.parent / "python")


def test_dist_parallel_multidevice():
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=8",
               JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [PYTHON_DIR] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.run([sys.executable, __file__], env=env,
                          capture_output=True, text=True, timeout=1200)
    report = proc.stdout + proc.stderr
    assert proc.returncode == 0 and "ALL PASS" in proc.stdout, report


def run_validation():
    from functools import partial

    import numpy as onp
    import jax
    import jax.numpy as jnp
    from jax import lax, shard_map
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax_md import space

    from lammps_jax.dist.parallel import feature_exchange
    from lammps_jax.dist.parallel import force_decomposition
    from lammps_jax.dist.parallel import ghost_exchange
    from lammps_jax.dist.parallel import local_neighbor_list
    from lammps_jax.dist.parallel import replicate_data

    CUTOFF = 2.5
    N_CELLS = 6
    L = N_CELLS * 1.5
    failures = []

    def check(name, ok, detail=""):
        print(("PASS" if ok else "FAIL"), name, detail)
        if not ok:
            failures.append(name)

    rng = onp.random.default_rng(11)
    basis = onp.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = onp.array([[i, j, k] for i in range(N_CELLS) for j in range(N_CELLS) for k in range(N_CELLS)])
    R = ((cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) / N_CELLS)
    R = (R + rng.uniform(-0.01, 0.01, R.shape)) % 1.0
    N = R.shape[0]
    box = jnp.eye(3) * L
    # triclinic box, lattice vectors as columns
    box_triclinic = jnp.asarray([[L, 0.25 * L, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]])
    species_global = (onp.arange(N) % 5 + 1).astype(onp.int32)

    def lj(dist):
        inv6 = dist ** -6
        return 4.0 * (inv6 ** 2 - inv6)

    def dense_reference(positions, box):
        def total(pos):
            delta = pos[:, None, :] - pos[None, :, :]
            delta = delta - jnp.round(delta)
            dr = jnp.sqrt(jnp.sum((delta @ box.T) ** 2, -1) + jnp.eye(N))
            mask = (dr < CUTOFF) & ~jnp.eye(N, dtype=bool)
            return 0.5 * jnp.sum(jnp.where(mask, lj(jnp.where(mask, dr, 1.0)), 0.0))
        E, gradient = jax.value_and_grad(total)(jnp.asarray(positions))
        return (float(E), -onp.asarray(gradient))

    E_ref, F_ref = dense_reference(R, box)

    out, n_out = ghost_exchange.append_rows(
        jnp.zeros((6, 2)), jnp.int32(0),
        jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0, 0], [0, 0]]),
        jnp.int32(3), 6)
    check("append_rows", int(n_out) == 3 and
          onp.allclose(onp.asarray(out[:3]), [[1, 2], [3, 4], [5, 6]]))

    # one ulp above a tile multiple gains a hop
    config_ulp = ghost_exchange.create_config(20, 1000, jnp.eye(3), 0.45000000000000007, grid=(20, 1, 1))
    check("hops ulp guard", config_ulp.hops[0] == 10, f"hops={config_ulp.hops}")

    lattice = onp.asarray(box_triclinic)
    normals = [onp.cross(lattice[:, (k + 1) % 3], lattice[:, (k + 2) % 3]) for k in range(3)]
    widths = onp.array([abs(onp.dot(lattice[:, 0], normals[0])) / onp.linalg.norm(normal)
                        for normal in normals])
    config_triclinic = ghost_exchange.create_config(8, N, box_triclinic, CUTOFF, grid=(2, 2, 2))
    check("triclinic cutoff_frac",
          onp.allclose(config_triclinic.cutoff_frac, CUTOFF / widths)
          and config_triclinic.cutoff_frac[0] > CUTOFF / L,
          f"cutoff_frac={config_triclinic.cutoff_frac}")

    def owned_buffers(grid):
        """(owned (n_ranks, capacity, 5) [pos, species, id], counts) by geometric tile."""
        n_ranks = grid[0] * grid[1] * grid[2]
        rank_of = onp.zeros(N, int)
        for axis in range(3):
            coord = onp.clip((R[:, axis] * grid[axis]).astype(int), 0, grid[axis] - 1)
            rank_of = rank_of * grid[axis] + coord
        counts = onp.bincount(rank_of, minlength=n_ranks).astype(onp.int32)
        owned = onp.zeros((n_ranks, int(counts.max()), 5))
        owned[:, :, 3] = 99
        for domain in range(n_ranks):
            selected = onp.where(rank_of == domain)[0]
            owned[domain, :len(selected), :3] = R[selected]
            owned[domain, :len(selected), 3] = species_global[selected]
            owned[domain, :len(selected), 4] = selected
        return owned, counts

    def scatter_owned(F_owned, owned, counts):
        """(N, 3) global forces from per-rank owned forces via the id column."""
        F = onp.zeros((N, 3))
        for domain in range(owned.shape[0]):
            F[owned[domain, :counts[domain], 4].astype(int)] = onp.asarray(F_owned[domain, :counts[domain]])
        return F

    def ghost_setup(grid, box, config, owned):
        cells_per_side, cell_capacity, max_neighbors, max_edges = ghost_exchange.subgraph_params(
            config, box, N, capacity_mult=2.5)
        wrap = jnp.asarray([1.0 if ranks == 1 else 0.0 for ranks in grid])

        def build(owned_local, n_owned_local):
            ghost, n_ghost, exchange_flags, plan = ghost_exchange.ghost_exchange(owned_local, n_owned_local, config, "i")
            node_idx, senders, receivers, species, edge_mask, overflow_graph = \
                ghost_exchange.ghost_exchange_subgraph(
                    owned_local[:, :3], ghost[:, :3],
                    owned_local[:, 3].astype(jnp.int32), ghost[:, 3].astype(jnp.int32),
                    owned_local[:, 4].astype(jnp.int32), ghost[:, 4].astype(jnp.int32),
                    n_owned_local, n_ghost, box, cells_per_side, cell_capacity, max_neighbors, max_edges, config, "i")
            flags = (jnp.any(exchange_flags) | overflow_graph).astype(jnp.int32)
            return plan, node_idx, senders, receivers, species, edge_mask, flags

        return build, wrap

    def grid_case(grid, box, send_equals_ghost=False, tag="", max_ghost=None, dtype=None):
        n_ranks = grid[0] * grid[1] * grid[2]
        E_ref_box, F_ref_box = dense_reference(R, box)
        owned, counts = owned_buffers(grid)
        config = ghost_exchange.create_config(n_ranks, N, box, CUTOFF, capacity_mult=1.5, grid=grid)
        label = f"grid{grid}{tag}"
        tol_e, tol_f = (1e-10, 1e-9)
        if dtype is not None:
            owned, box = (owned.astype(dtype), jnp.asarray(box, dtype=dtype))
            tol_e, tol_f = (1e-6, 3e-5 * float(onp.max(onp.abs(F_ref_box))))
            label += f" {onp.dtype(dtype).name}"
        if send_equals_ghost:
            config = ghost_exchange.GhostExchangeConfig(*config[:4], config.max_ghost, *config[5:])
            label += " send=ghost"
        if max_ghost is not None:
            config = ghost_exchange.GhostExchangeConfig(*config[:3], max_ghost, *config[4:])
            label += f" max_ghost={max_ghost}"
        build, wrap = ghost_setup(grid, box, config, owned)
        if dtype is not None:
            wrap = jnp.asarray(wrap, dtype)
        mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))

        @partial(shard_map, mesh=mesh, in_specs=(P("i"), P("i")),
                 out_specs=(P(), P("i"), P(), P("i"), P("i"), P("i"), P("i"), P("i")),
                 check_vma=False)
        def grid_ef(owned_stack, n_owned_arr):
            owned_local = owned_stack.squeeze(0)
            plan, node_idx, senders, receivers, species, edge_mask, flags = build(owned_local, n_owned_arr[0])

            def local_energy(owned_pos):
                pos = ghost_exchange.exchange_apply(plan, owned_pos, config, "i", unwrap=True)
                delta = pos[receivers] - pos[senders]
                delta = delta - jnp.round(delta) * wrap
                dist_sq = jnp.sum((delta @ box.T) ** 2, -1)
                mask = edge_mask & (dist_sq > 1e-12)
                dist = jnp.sqrt(jnp.where(mask, dist_sq, 1.0))
                return 0.5 * jnp.sum(jnp.where(mask, lj(dist), 0.0))

            E_local, gradient = jax.value_and_grad(local_energy)(owned_local[:, :3])
            return (lax.psum(E_local, "i"), -gradient[None], lax.psum(flags, "i"),
                    node_idx[None], species[None], senders[None], receivers[None], edge_mask[None])

        shard = NamedSharding(mesh, P("i"))
        E, F_owned, flags, node_idx_all, species_all, senders_all, receivers_all, edge_mask_all = grid_ef(
            jax.device_put(jnp.asarray(owned), shard), jax.device_put(jnp.asarray(counts), shard))
        e_rel = abs(float(E) - E_ref_box) / abs(E_ref_box)
        f_abs = float(onp.max(onp.abs(scatter_owned(F_owned, owned, counts) - F_ref_box)))
        species_ok = True
        for domain in range(n_ranks):
            nodes_domain = onp.asarray(node_idx_all[domain])
            species_domain = onp.asarray(species_all[domain])
            senders_domain = onp.asarray(senders_all[domain])
            receivers_domain = onp.asarray(receivers_all[domain])
            edge_mask_domain = onp.asarray(edge_mask_all[domain])
            live = onp.unique(onp.concatenate([senders_domain[edge_mask_domain], receivers_domain[edge_mask_domain]]))
            species_ok &= bool(onp.all(species_domain[live] == species_global[nodes_domain[live]]))
            species_ok &= bool(onp.all((species_domain == 0) | (species_domain == species_global[nodes_domain])))
        passed = e_rel < tol_e and f_abs < tol_f and int(flags) == 0 and species_ok
        if max_ghost is not None:
            passed = int(flags) > 0
        check(label, passed,
              f"hops={config.hops} e_rel={e_rel:.1e} f_abs={f_abs:.1e} "
              f"flags={int(flags)} species_ok={species_ok}")

    for grid in [(8, 1, 1), (4, 2, 1), (2, 2, 2), (1, 4, 2), (3, 2, 1), (1, 1, 1)]:
        grid_case(grid, box)
    grid_case((2, 2, 2), box, send_equals_ghost=True)
    grid_case((2, 2, 2), box, max_ghost=4)
    grid_case((2, 2, 2), box_triclinic, tag=" triclinic")
    grid_case((2, 2, 2), box, dtype=onp.float32)

    # two message-passing layers over the recorded plan against the dense graph
    def toy_node_energies(dR, edge_mask, species_owned, senders, receivers, exchange_fn):
        rows = species_owned.shape[0]
        features = jax.nn.one_hot(species_owned, 6)
        weight = jnp.where(edge_mask, jnp.exp(-jnp.sqrt(jnp.sum(dR ** 2, -1) + ~edge_mask)), 0.0)
        for layer in range(2):
            messages = exchange_fn(features)[senders] * weight[:, None]
            aggregated = jnp.zeros((rows, 6)).at[receivers].add(messages, mode="drop")
            features = jnp.tanh(features + 0.1 * aggregated)
        return jnp.sum(features ** 2, axis=1)

    delta_all = R[:, None, :] - R[None, :, :]
    delta_all -= onp.round(delta_all)
    dist_all = onp.sqrt(onp.sum((delta_all @ onp.asarray(box).T) ** 2, -1))
    toy_senders, toy_receivers = onp.nonzero((dist_all < CUTOFF) & ~onp.eye(N, dtype=bool))

    def dense_toy(pos):
        delta = pos[toy_receivers] - pos[toy_senders]
        delta = delta - jnp.round(delta)
        return jnp.sum(toy_node_energies(space.transform(box, delta), jnp.ones(len(toy_senders), bool), species_global,
                                         toy_senders, toy_receivers, lambda features: features))

    E_toy_ref, toy_gradient = jax.value_and_grad(dense_toy)(jnp.asarray(R))
    F_toy_ref = -onp.asarray(toy_gradient)

    def toy_case(grid, skin=0.0):
        n_ranks = grid[0] * grid[1] * grid[2]
        owned, counts = owned_buffers(grid)
        config = ghost_exchange.create_config(n_ranks, N, box, CUTOFF, capacity_mult=1.5, grid=grid, skin=skin)
        build = ghost_setup(grid, box, config, owned)[0]
        mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))

        @partial(shard_map, mesh=mesh, in_specs=(P("i"), P("i")), out_specs=(P(), P("i"), P()), check_vma=False)
        def toy_ef(owned_stack, n_owned_arr):
            owned_local = owned_stack.squeeze(0)
            n_owned_local = n_owned_arr[0]
            plan, node_idx, senders, receivers, species, edge_mask, flags = build(owned_local, n_owned_local)
            E_local, gradient = jax.value_and_grad(lambda pos: feature_exchange.ghost_energy(
                toy_node_energies, pos, n_owned_local, owned_local[:, 3].astype(jnp.int32), senders, receivers,
                edge_mask, plan, box, config, "i"))(owned_local[:, :3])
            return lax.psum(E_local, "i"), -gradient[None], lax.psum(flags, "i")

        shard = NamedSharding(mesh, P("i"))
        E, F_owned, flags = toy_ef(jax.device_put(jnp.asarray(owned), shard), jax.device_put(jnp.asarray(counts), shard))
        e_rel = abs(float(E) - float(E_toy_ref)) / abs(float(E_toy_ref))
        f_abs = float(onp.max(onp.abs(scatter_owned(F_owned, owned, counts) - F_toy_ref)))
        check(f"per-layer exchange grid{grid}" + (f" skin={skin}" if skin else ""),
              e_rel < 1e-10 and f_abs < 1e-9 and int(flags) == 0,
              f"e_rel={e_rel:.1e} f_abs={f_abs:.1e} flags={int(flags)}")

    for grid in [(8, 1, 1), (2, 2, 2)]:
        toy_case(grid)
    # the toy energy has no cutoff envelope, so ghost_energy must mask the skin shell itself
    toy_case((2, 2, 2), skin=0.3)

    # force decomposition on a 2 x 2 mesh; jax_md forces are real-space, reference divided by L
    displacement = space.periodic_general(box, fractional_coordinates=True)[0]
    pair_fn = lambda dist: jnp.where(dist < CUTOFF, lj(dist), 0.0)
    fd_config = force_decomposition.create_config(N, 4)
    E_fd, F_fd = force_decomposition.make_sharded_energy_force(pair_fn, displacement, fd_config)(jnp.asarray(R))
    e_rel = abs(float(E_fd) - E_ref) / abs(E_ref)
    f_abs = float(onp.max(onp.abs(onp.asarray(F_fd) - F_ref / L)))
    check("force decomposition 2x2", e_rel < 1e-10 and f_abs < 1e-9, f"e_rel={e_rel:.1e} f_abs={f_abs:.1e}")
    for bad_atoms, bad_devices in ((N, 8), (N + 1, 4)):
        try:
            force_decomposition.create_config(bad_atoms, bad_devices)
            check(f"force decomposition raise {bad_atoms} {bad_devices}", False)
        except ValueError:
            check(f"force decomposition raise {bad_atoms} {bad_devices}", True)

    # feature_exchange against the dense reference
    def feature_exchange_case(grid):
        n_ranks = 8
        n_per = N // n_ranks
        perm, inverse = feature_exchange.grid_order(R, grid)
        tile = [onp.clip((R[:, axis_index] % 1.0 * grid[axis_index]).astype(int), 0, grid[axis_index] - 1)
                for axis_index in range(3)]
        rank = (tile[0] * grid[1] + tile[1]) * grid[2] + tile[2]
        ordered = bool(onp.all(onp.diff(rank[perm]) >= 0)) and \
            bool(onp.all(perm[inverse] == onp.arange(N)))
        R_permuted = R[perm]
        d_permuted = R_permuted[:, None, :] - R_permuted[None, :, :]
        d_permuted -= onp.round(d_permuted)
        dist_permuted = onp.sqrt(onp.sum((d_permuted @ onp.asarray(box).T) ** 2, -1))
        senders_global, receivers_global = onp.nonzero((dist_permuted < CUTOFF) & ~onp.eye(N, dtype=bool))
        (domain_node_idx, domain_senders, domain_receivers, domain_shifts,
         domain_species, domain_edge_mask) = feature_exchange.precompute_subgraphs(
            jnp.asarray(senders_global, jnp.int32), jnp.asarray(receivers_global, jnp.int32),
            jnp.zeros((len(senders_global), 3)), jnp.ones(len(senders_global), bool),
            N, n_ranks, 1, jnp.zeros(N, jnp.int32))[:6]
        config = replicate_data.create_config(n_domains=n_ranks, n_atoms_per_domain=n_per)
        mesh = config.mesh
        axis = config.axis_name
        sharding = NamedSharding(mesh, P(axis))
        exchange = feature_exchange.make_exchange(N, n_per, n_ranks, axis)

        @partial(shard_map, mesh=mesh, in_specs=(P(axis),) * 5,
                 out_specs=(P(), P(axis)), check_vma=False)
        def feature_exchange_ef(R_local, node_idx, local_senders, local_receivers, edge_mask):
            node_idx, local_senders, local_receivers, edge_mask = (
                array.squeeze(0) for array in (node_idx, local_senders, local_receivers, edge_mask))
            # dynamic_update_slice needs int64 indices under x64
            atom_start = lax.convert_element_type(lax.axis_index(axis) * n_per, jnp.int64)

            def block_energy(R_owned):
                R_all = exchange(R_owned, atom_start)
                pos = R_all[node_idx]
                dr_frac = pos[local_receivers] - pos[local_senders]
                dr_frac = dr_frac - jnp.round(dr_frac)
                dist_sq = jnp.sum((dr_frac @ box.T) ** 2, -1)
                valid = edge_mask & (dist_sq > 1e-12) & (local_receivers < n_per)
                dist = jnp.sqrt(jnp.where(valid, dist_sq, 1.0))
                return 0.5 * jnp.sum(jnp.where(valid, lj(dist), 0.0))

            E_local, grad_owned = jax.value_and_grad(block_energy)(R_local.squeeze(0))
            return lax.psum(E_local, axis), -grad_owned[None]

        E, F_owned = feature_exchange_ef(
            jax.device_put(jnp.asarray(R_permuted).reshape(n_ranks, n_per, 3), sharding),
            jax.device_put(domain_node_idx, sharding), jax.device_put(domain_senders, sharding),
            jax.device_put(domain_receivers, sharding), jax.device_put(domain_edge_mask, sharding))
        F = onp.asarray(F_owned.reshape(-1, 3))[inverse]
        e_rel = abs(float(E) - E_ref) / abs(E_ref)
        f_abs = float(onp.max(onp.abs(F - F_ref)))
        check(f"feature grid{grid}", e_rel < 1e-10 and f_abs < 1e-9 and ordered,
              f"nodes={domain_node_idx.shape[1]} edges={domain_senders.shape[1]} "
              f"e_rel={e_rel:.1e} f_abs={f_abs:.1e} ordered={ordered}")

    for grid in [(8, 1, 1), (2, 2, 2), (1, 4, 2)]:
        feature_exchange_case(grid)

    # redistribute across the periodic seam with species and id columns
    ids_global = onp.arange(N, dtype=onp.int32)
    int_columns = onp.stack([species_global, ids_global], 1)

    def redistribute_case(grid, shift, expect_misplaced=False, max_send=None, tight=False):
        n_ranks = grid[0] * grid[1] * grid[2]
        config = ghost_exchange.create_config(n_ranks, N, box, CUTOFF, capacity_mult=2.0, grid=grid)
        if tight:
            config = config._replace(max_owned=int(ghost_exchange.pack_tiles(R, config)[2].max()))
        if max_send is not None:
            config = config._replace(max_send=max_send)
        owned, owned_ints, counts, rank_of = ghost_exchange.pack_tiles(R, config, ints=int_columns)
        # a tight buffer overflows when only the lower half moves, so one tile gains without losing
        disp = onp.where(R[:, :1] < 0.5, onp.asarray(shift), 0.0) if tight else onp.broadcast_to(onp.asarray(shift), R.shape)
        shifted = (R + disp) % 1.0
        owned = (owned + disp[owned_ints[:, :, 1]]) % 1.0
        expect_owned_overflow = tight and int(ghost_exchange.pack_tiles(
            shifted, config._replace(max_owned=N))[2].max()) > config.max_owned
        mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))

        @partial(shard_map, mesh=mesh, in_specs=(P("i"), P("i"), P("i")),
                 out_specs=(P("i"), P("i"), P("i"), P()), check_vma=False)
        def migrate(stack, ints_stack, n_arr):
            new_pos, new_ints, new_n, flags = ghost_exchange.redistribute(
                stack.squeeze(0), n_arr[0], config, "i", owned_ints=ints_stack.squeeze(0))
            return (new_pos[None], new_ints[None], jnp.asarray(new_n, jnp.int32)[None],
                    lax.psum(flags.astype(jnp.int32), "i"))

        shard = NamedSharding(mesh, P("i"))
        new_pos, new_ints, new_n, flags = jax.jit(migrate)(
            jax.device_put(jnp.asarray(owned), shard), jax.device_put(jnp.asarray(owned_ints), shard),
            jax.device_put(jnp.asarray(counts), shard))
        new_pos, new_ints, new_n, flags = (onp.asarray(array) for array in (new_pos, new_ints, new_n, flags))
        if expect_misplaced:
            check(f"redistribute{grid} misplaced-flag", flags[0] > 0 and flags[1] == 0 and flags[2] == 0,
                  f"flags={flags.tolist()}")
            return
        if max_send is not None or tight:
            check(f"redistribute{grid} capacity flags max_send={max_send} tight={tight}",
                  flags[0] == 0 and (flags[1] > 0) == (max_send is not None) and (flags[2] > 0) == expect_owned_overflow,
                  f"flags={flags.tolist()} expect_owned_overflow={expect_owned_overflow}")
            return
        placed = True
        by_id = onp.full((N, 3), onp.nan)
        for domain in range(n_ranks):
            live = new_pos[domain, :new_n[domain]]
            live_ids = new_ints[domain, :new_n[domain], 1]
            by_id[live_ids] = live
            code = domain
            for axis in (2, 1, 0):
                coord = code % grid[axis]
                code //= grid[axis]
                lo, hi = coord / grid[axis], (coord + 1) / grid[axis]
                placed &= bool(onp.all((live[:, axis] >= lo) & (live[:, axis] < hi)))
            placed &= bool(onp.all(new_ints[domain, :new_n[domain], 0] == species_global[live_ids]))
        conserved = int(new_n.sum()) == N and onp.allclose(by_id, shifted)
        check(f"redistribute{grid}", placed and conserved and int(flags.sum()) == 0,
              f"counts_sum={int(new_n.sum())} flags={flags.tolist()}")

    redistribute_case((2, 2, 2), (0.1, 0.07, 0.04), expect_misplaced=False)
    redistribute_case((4, 2, 1), (-0.06, 0.12, 0.0), expect_misplaced=False)
    redistribute_case((8, 1, 1), (0.3, 0.0, 0.0), expect_misplaced=True)
    redistribute_case((2, 2, 2), (0.1, 0.07, 0.04), max_send=4)
    redistribute_case((2, 2, 2), (0.1, 0.07, 0.04), tight=True)

    # Verlet skin: the plan recorded on R gives the reference energy and forces on positions displaced
    # by less than skin / 2,
    # and the trigger fires once an atom moves further
    def skin_case(grid, skin, disp, tag, expect_rebuild):
        n_ranks = grid[0] * grid[1] * grid[2]
        config = ghost_exchange.create_config(n_ranks, N, box, CUTOFF, capacity_mult=1.5, grid=grid, skin=skin)
        cells_per_side, cell_capacity, max_neighbors, max_edges = ghost_exchange.subgraph_params(
            config, box, N, capacity_mult=2.5)
        owned, owned_ints, counts, rank_of = ghost_exchange.pack_tiles(R, config, ints=int_columns)
        moved = onp.where(owned_ints[:, :, 1:2] >= 0, owned + disp[owned_ints[:, :, 1]], 0.0)
        wrap = jnp.asarray([1.0 if ranks == 1 else 0.0 for ranks in grid])
        mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))
        rows = config.max_owned

        @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 4, out_specs=(P(), P("i"), P(), P()), check_vma=False)
        def replay_ef(owned_stack, ints_stack, counts_arr, moved_stack):
            pos0, ints_local, n_owned, pos1 = (owned_stack.squeeze(0), ints_stack.squeeze(0), counts_arr[0],
                                               moved_stack.squeeze(0))
            ghost, n_ghost, exchange_flags, plan = ghost_exchange.ghost_exchange(pos0, n_owned, config, "i")
            ghost_ints = lax.dynamic_slice_in_dim(ghost_exchange.exchange_apply(plan, ints_local, config, "i"),
                                                  n_owned, config.max_ghost, axis=0)
            node_idx, senders, receivers, node_species, edge_mask, overflow_graph = ghost_exchange.ghost_exchange_subgraph(
                pos0, ghost, ints_local[:, 0], ghost_ints[:, 0], ints_local[:, 1], ghost_ints[:, 1], n_owned, n_ghost,
                box, cells_per_side, cell_capacity, max_neighbors, max_edges, config, "i")
            live_nodes = jnp.arange(node_idx.shape[0]) < n_owned + n_ghost
            pool = ghost_exchange.exchange_apply(plan, pos0, config, "i", unwrap=True)
            image_err = jnp.abs((pool[:, :3] - R_ref[node_idx]) % 1.0)
            image_err = jnp.max(jnp.minimum(image_err, 1.0 - image_err), axis=1)
            ids_ok = jnp.all(jnp.where(live_nodes, image_err < 1e-9, True))
            ids_ok = ids_ok & jnp.all(jnp.where(live_nodes, node_species == species_ref[node_idx], True))


            def local_energy(owned_pos):
                pos = ghost_exchange.exchange_apply(plan, owned_pos, config, "i", unwrap=True)
                delta = pos[receivers] - pos[senders]
                delta = delta - jnp.round(delta) * wrap
                dist_sq = jnp.sum((delta @ box.T) ** 2, -1)
                mask = edge_mask & (dist_sq > 1e-12) & (dist_sq < CUTOFF ** 2)
                return 0.5 * jnp.sum(jnp.where(mask, lj(jnp.sqrt(jnp.where(mask, dist_sq, 1.0))), 0.0))

            E_local, gradient = jax.value_and_grad(local_energy)(pos1)
            rebuild = ghost_exchange.needs_rebuild(pos1, pos0, n_owned, box, config, "i")
            flags = lax.psum((jnp.any(exchange_flags) | overflow_graph | ~ids_ok).astype(jnp.int32), "i")
            return lax.psum(E_local, "i"), -gradient[None], flags, rebuild

        R_ref = jnp.asarray(R)
        species_ref = jnp.asarray(species_global)
        shard = NamedSharding(mesh, P("i"))
        E, F_owned, flags, rebuild = jax.jit(replay_ef)(*(jax.device_put(jnp.asarray(array), shard)
                                                           for array in (owned, owned_ints, counts, moved)))
        E_moved, F_moved = dense_reference((R + disp) % 1.0, box)
        F = onp.zeros((N, 3))
        for domain in range(n_ranks):
            F[owned_ints[domain, :counts[domain], 1]] = onp.asarray(F_owned[domain, :counts[domain]])
        e_rel = abs(float(E) - E_moved) / abs(E_moved)
        f_abs = float(onp.max(onp.abs(F - F_moved)))
        exact = e_rel < 1e-10 and f_abs < 1e-9
        check(f"skin replay{grid}{tag}", int(flags) == 0 and bool(rebuild) == expect_rebuild and (exact or expect_rebuild),
              f"hops={config.hops} e_rel={e_rel:.1e} f_abs={f_abs:.1e} flags={int(flags)} rebuild={bool(rebuild)}")

    small = rng.uniform(-0.009, 0.009, (N, 3))
    for grid in [(8, 1, 1), (2, 2, 2), (1, 4, 2)]:
        skin_case(grid, 0.3, small, "", expect_rebuild=False)
    skin_case((2, 2, 2), 0.3, rng.uniform(-0.03, 0.03, (N, 3)), " too far", expect_rebuild=True)
    # a wrapped coordinate must force a rebuild even though the physical move is tiny
    wrapped = onp.where(R[:, :1] + small[:, :1] >= 1.0, small - onp.array([1.0, 0.0, 0.0]), small)
    wrapped = onp.where(R[:, :1] + small[:, :1] < 0.0, small + onp.array([1.0, 0.0, 0.0]), wrapped)
    skin_case((2, 2, 2), 0.3, wrapped, " wrapped", expect_rebuild=True)

    # needs_rebuild fires just past skin / 2 and ignores dead rows
    def rebuild_probe(config, reference, moved, counts):
        mesh = Mesh(jax.devices()[:config.n_ranks], axis_names=("i",))
        shard = NamedSharding(mesh, P("i"))
        probe = shard_map(lambda ref, cur, n: ghost_exchange.needs_rebuild(cur.squeeze(0), ref.squeeze(0), n[0],
                                                                           box, config, "i"),
                          mesh=mesh, in_specs=(P("i"),) * 3, out_specs=P(), check_vma=False)
        return bool(jax.jit(probe)(*(jax.device_put(jnp.asarray(a), shard) for a in (reference, moved, counts))))

    config_skin = ghost_exchange.create_config(8, N, box, CUTOFF, grid=(2, 2, 2), skin=0.3)
    owned_skin, _, counts_skin, _ = ghost_exchange.pack_tiles(R, config_skin)
    outcomes = []
    for eps in (1e-6, -1e-6):
        moved = owned_skin.copy()
        moved[3, 0, 0] += (0.15 + eps) / L
        outcomes.append(rebuild_probe(config_skin, owned_skin, moved, counts_skin))
    dead = owned_skin.copy()
    dead[3, counts_skin[3], 0] += 10.0
    outcomes.append(rebuild_probe(config_skin, owned_skin, dead, counts_skin))
    check("needs_rebuild boundary", outcomes == [True, False, False], f"outcomes={outcomes}")

    # every flag fires: a live row outside its tile, then each capacity in turn
    def flag_probe(config, cells_per_side, cell_capacity, max_neighbors, max_edges, offset=0.0):
        n_ranks = config.n_ranks
        owned, owned_ints, counts, rank_of = ghost_exchange.pack_tiles(R, config, ints=int_columns)
        owned = onp.where(owned_ints[:, :, 1:2] >= 0, owned + onp.array([offset, 0.0, 0.0]), 0.0)
        mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))

        @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 3, out_specs=P(), check_vma=False)
        def probe(stack, ints_stack, counts_arr):
            pos, ints, n_owned = (stack.squeeze(0), ints_stack.squeeze(0), counts_arr[0])
            ghost, n_ghost, exchange_flags, plan = ghost_exchange.ghost_exchange(pos, n_owned, config, "i")
            ghost_ints = lax.dynamic_slice_in_dim(ghost_exchange.exchange_apply(plan, ints, config, "i"),
                                                  n_owned, config.max_ghost, axis=0)
            overflow_graph = ghost_exchange.ghost_exchange_subgraph(
                pos, ghost, ints[:, 0], ghost_ints[:, 0], ints[:, 1], ghost_ints[:, 1], n_owned, n_ghost,
                box, cells_per_side, cell_capacity, max_neighbors, max_edges, config, "i")[5]
            return lax.psum(jnp.concatenate([exchange_flags, overflow_graph[None]]).astype(jnp.int32), "i")

        shard = NamedSharding(mesh, P("i"))
        return onp.asarray(jax.jit(probe)(*(jax.device_put(jnp.asarray(a), shard)
                                            for a in (owned, owned_ints, counts)))).tolist()

    config_flags = ghost_exchange.create_config(8, N, box, CUTOFF, capacity_mult=1.5, grid=(2, 2, 2))
    params_flags = ghost_exchange.subgraph_params(config_flags, box, N, capacity_mult=2.5)
    outcomes = {
        "baseline": flag_probe(config_flags, *params_flags),
        "misplaced": flag_probe(config_flags, *params_flags, offset=0.02),
        "send": flag_probe(config_flags._replace(max_send=8), *params_flags),
        "cell": flag_probe(config_flags, params_flags[0], 1, *params_flags[2:]),
        "neighbors": flag_probe(config_flags, *params_flags[:2], 4, params_flags[3]),
        "edges": flag_probe(config_flags, *params_flags[:3], 16),
    }
    check("flags fire",
          outcomes["baseline"] == [0, 0, 0] and outcomes["misplaced"][0] > 0 and outcomes["misplaced"][1] == 0
          and outcomes["send"][0] == 0 and outcomes["send"][1] > 0 and outcomes["cell"][2] > 0
          and outcomes["neighbors"][2] > 0 and outcomes["edges"][2] > 0,
          " ".join(f"{name}={value}" for name, value in outcomes.items()))

    # guards: int32 counts, unwrap on integers, a cell grid finer than the reach
    count_dtype = ghost_exchange.pack_selected(jnp.arange(5), 5, jnp.ones(5, bool), 4)[1].dtype
    try:
        ghost_exchange.exchange_apply((), jnp.zeros((4, 2), jnp.int32), config_skin, "i", unwrap=True)
        unwrap_raises = False
    except TypeError:
        unwrap_raises = True
    params_skin = ghost_exchange.subgraph_params(config_skin, box, N)
    try:
        ghost_exchange.ghost_exchange_subgraph(
            jnp.zeros((4, 3)), jnp.zeros((8, 3)), jnp.zeros(4, jnp.int32), jnp.zeros(8, jnp.int32),
            jnp.zeros(4, jnp.int32), jnp.zeros(8, jnp.int32), 4, 0, box, params_skin[0] * 3, *params_skin[1:],
            config_skin, "i")
        cells_raise = False
    except ValueError:
        cells_raise = True
    fine = bool(local_neighbor_list.cells_too_fine(box, onp.array([3, 3, 12]), CUTOFF))
    coarse = bool(local_neighbor_list.cells_too_fine(box, onp.array([3, 3, 3]), CUTOFF))
    check("guards", count_dtype == jnp.int32 and unwrap_raises and cells_raise and fine and not coarse,
          f"count_dtype={count_dtype} unwrap_raises={unwrap_raises} cells_raise={cells_raise} "
          f"fine={fine} coarse={coarse}")

    # an MD-style loop: drift unwrapped, replay while inside the skin, redistribute and rebuild otherwise
    def loop_case(grid, skin=0.3, n_steps=8, step=0.006):
        n_ranks = grid[0] * grid[1] * grid[2]
        config = ghost_exchange.create_config(n_ranks, N, box, CUTOFF, capacity_mult=1.5, grid=grid, skin=skin)
        cells_per_side, cell_capacity, max_neighbors, max_edges = ghost_exchange.subgraph_params(
            config, box, N, capacity_mult=2.5)
        wrap = jnp.asarray([1.0 if ranks == 1 else 0.0 for ranks in grid])
        mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))
        shard = NamedSharding(mesh, P("i"))

        @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 3, out_specs=(P("i"),) * 7 + (P(),), check_vma=False)
        def rebuild(stack, ints_stack, counts_arr):
            pos, ints, n_owned, migrate_flags = ghost_exchange.redistribute(
                stack.squeeze(0), counts_arr[0], config, "i", owned_ints=ints_stack.squeeze(0))
            ghost, n_ghost, exchange_flags, plan = ghost_exchange.ghost_exchange(pos, n_owned, config, "i")
            ghost_ints = lax.dynamic_slice_in_dim(ghost_exchange.exchange_apply(plan, ints, config, "i"),
                                                  n_owned, config.max_ghost, axis=0)
            node_idx, senders, receivers, node_species, edge_mask, overflow = ghost_exchange.ghost_exchange_subgraph(
                pos, ghost, ints[:, 0], ghost_ints[:, 0], ints[:, 1], ghost_ints[:, 1], n_owned, n_ghost,
                box, cells_per_side, cell_capacity, max_neighbors, max_edges, config, "i")
            flags = lax.psum((jnp.any(migrate_flags) | jnp.any(exchange_flags) | overflow).astype(jnp.int32), "i")
            stacked = jax.tree.map(lambda a: a[None], (pos, ints, n_owned, plan, senders, receivers, edge_mask))
            return (*stacked, flags)

        @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 7, out_specs=(P(), P("i"), P()), check_vma=False)
        def energy_force(pos_stack, ref_stack, counts_arr, plan, senders, receivers, edge_mask):
            pos, ref, n_owned = (pos_stack.squeeze(0), ref_stack.squeeze(0), counts_arr[0])
            plan, senders, receivers, edge_mask = jax.tree.map(lambda a: a.squeeze(0),
                                                               (plan, senders, receivers, edge_mask))

            def local_energy(owned_pos):
                nodes = ghost_exchange.exchange_apply(plan, owned_pos, config, "i", unwrap=True)
                delta = nodes[receivers] - nodes[senders]
                delta = delta - jnp.round(delta) * wrap
                dist_sq = jnp.sum((delta @ box.T) ** 2, -1)
                mask = edge_mask & (dist_sq > 1e-12) & (dist_sq < CUTOFF ** 2)
                return 0.5 * jnp.sum(jnp.where(mask, lj(jnp.sqrt(jnp.where(mask, dist_sq, 1.0))), 0.0))

            E_local, gradient = jax.value_and_grad(local_energy)(pos)
            return (lax.psum(E_local, "i"), -gradient[None],
                    ghost_exchange.needs_rebuild(pos, ref, n_owned, box, config, "i"))

        rebuild = jax.jit(rebuild)
        energy_force = jax.jit(energy_force)
        owned, owned_ints, counts, _ = ghost_exchange.pack_tiles(R, config, ints=int_columns)
        pos, ints, counts_dev, *graph, flags = rebuild(*(jax.device_put(jnp.asarray(a), shard)
                                                         for a in (owned, owned_ints, counts)))
        ref = pos
        R_global = R.copy()
        n_rebuilds, n_replays, flag_total = (1, 0, int(flags))
        worst_e, worst_f, in_tile, conserved = (0.0, 0.0, True, True)
        for step_index in range(n_steps):
            disp = rng.uniform(-step, step, (N, 3))
            R_global = R_global + disp
            pos = pos + jnp.asarray(disp)[onp.asarray(ints)[:, :, 1]]
            E, F_owned, rebuild_flag = energy_force(pos, ref, counts_dev, *graph)
            if bool(rebuild_flag):
                pos, ints, counts_dev, *graph, flags = rebuild(pos, ints, counts_dev)
                ref = pos
                flag_total += int(flags)
                n_rebuilds += 1
                E, F_owned, rebuild_flag = energy_force(pos, ref, counts_dev, *graph)
                pos_host, ints_host, counts_host = (onp.asarray(a) for a in (pos, ints, counts_dev))
                for domain in range(n_ranks):
                    live = pos_host[domain, :counts_host[domain]]
                    code = domain
                    for axis in (2, 1, 0):
                        coord = code % grid[axis]
                        code //= grid[axis]
                        in_tile &= bool(onp.all((live[:, axis] >= coord / grid[axis])
                                                & (live[:, axis] < (coord + 1) / grid[axis])))
                ids = onp.concatenate([ints_host[d, :counts_host[d], 1] for d in range(n_ranks)])
                conserved &= int(counts_host.sum()) == N and len(onp.unique(ids)) == N
            else:
                n_replays += 1
            E_step, F_step = dense_reference(R_global % 1.0, box)
            ints_host, counts_host = (onp.asarray(ints), onp.asarray(counts_dev))
            F = onp.zeros((N, 3))
            for domain in range(n_ranks):
                F[ints_host[domain, :counts_host[domain], 1]] = onp.asarray(F_owned[domain, :counts_host[domain]])
            worst_e = max(worst_e, abs(float(E) - E_step) / abs(E_step))
            force_scale = max(1.0, float(onp.max(onp.abs(F_step))))
            worst_f = max(worst_f, float(onp.max(onp.abs(F - F_step))) / force_scale)
        wrapped_any = bool(onp.any((R_global < 0.0) | (R_global >= 1.0)))
        check(f"md loop grid{grid}",
              worst_e < 1e-10 and worst_f < 1e-12 and flag_total == 0 and in_tile and conserved
              and n_rebuilds >= 2 and n_replays >= 1 and wrapped_any,
              f"hops={config.hops} rebuilds={n_rebuilds} replays={n_replays} e_rel={worst_e:.1e} "
              f"f_rel={worst_f:.1e} flags={flag_total} in_tile={in_tile} conserved={conserved} wrapped={wrapped_any}")

    loop_case((2, 2, 2))
    loop_case((8, 1, 1))

    # the default grid has the fewest ghosts of all factorisations
    config_default = ghost_exchange.create_config(8, N, box, CUTOFF)
    fractions = {grid: ghost_exchange.halo_fractions(grid, config_default.cutoff_frac)[1]
                 for grid in [(8, 1, 1), (4, 2, 1), (2, 2, 2), (1, 4, 2), (1, 1, 8), (2, 4, 1)]}
    check("default grid", fractions[config_default.grid] <= min(fractions.values()) + 1e-12,
          f"grid={config_default.grid}")
    config_big = ghost_exchange.create_config(64, 10 ** 6, jnp.eye(3) * 271.0, 6.0)
    check("default grid cubic", config_big.grid == (4, 4, 4), f"grid={config_big.grid}")

    # local neighbor list and subgraph against the dense reference
    n_per = 128
    owned_start = 128
    delta = R[:, None, :] - R[None, :, :]
    delta -= onp.round(delta)
    dist = onp.sqrt(onp.sum((delta @ onp.asarray(box).T) ** 2, -1))
    cells_per_side, cell_capacity, max_neighbors = local_neighbor_list.estimate_cell_params(
        box, CUTOFF, N, capacity_mult=2.0)
    neighbor_idx, overflow = jax.jit(lambda pos: local_neighbor_list.local_neighbor_list(
        pos, box, CUTOFF, owned_start, n_per, cells_per_side, cell_capacity, max_neighbors))(jnp.asarray(R))
    got = {(int(sender), int(receiver)) for sender, receiver
           in zip(onp.asarray(neighbor_idx[0]), onp.asarray(neighbor_idx[1])) if receiver < N}
    want = {(i, j) for i in range(owned_start, owned_start + n_per)
            for j in range(N) if j != i and dist[i, j] < CUTOFF}
    check("local_neighbor_list", got == want and not bool(overflow),
          f"edges={len(got)} ref={len(want)}")

    cells_per_side, cell_capacity, max_nodes, max_edges_per_atom = local_neighbor_list.estimate_subgraph_params(
        box, CUTOFF, N, n_per, 1, capacity_mult=2.0)
    node_idx, senders, receivers, shifts, species, edge_mask, overflow = jax.jit(
        lambda pos: local_neighbor_list.local_subgraph(
            pos, jnp.zeros(N, jnp.int32), box, CUTOFF, owned_start, n_per,
            1, cells_per_side, cell_capacity, max_nodes, max_edges_per_atom))(jnp.asarray(R))
    frac_nodes = jnp.concatenate([jnp.asarray(R)[node_idx], jnp.zeros((1, 3))])
    d_frac = frac_nodes[receivers] - (frac_nodes[senders] + shifts)
    dr = jnp.sqrt(jnp.sum((d_frac @ box.T) ** 2, -1) + (~edge_mask).astype(frac_nodes.dtype))
    E_sub = float(0.5 * jnp.sum(jnp.where(edge_mask, lj(jnp.where(edge_mask, dr, 1.0)), 0.0)))
    block = dist[owned_start:owned_start + n_per]
    block_mask = (block < CUTOFF) & (block > 0)
    block_safe = onp.where(block_mask, block, 1.0)
    E_block = float(0.5 * onp.sum(onp.where(
        block_mask, 4.0 * (block_safe ** -12 - block_safe ** -6), 0.0)))
    check("local_subgraph shifts", abs(E_sub - E_block) < 1e-9 * abs(E_block) and not bool(overflow),
          f"E={E_sub:.6f} ref={E_block:.6f}")

    try:
        feature_exchange.precompute_subgraphs(
            jnp.asarray([0, 1, 2, 3], jnp.int32), jnp.asarray([1, 2, 3, 0], jnp.int32),
            jnp.zeros((4, 3)), jnp.ones(4, bool), 4, 2, 1, jnp.zeros(4, jnp.int32),
            max_nodes=2)
        check("precompute_subgraphs raise", False)
    except ValueError:
        check("precompute_subgraphs raise", True)

    pair = lambda dist: jnp.where(dist < CUTOFF, jnp.exp(-dist), 0.0)
    domain_energy = replicate_data.domain_energy_from_pair_with_neighbors(pair, displacement)
    # neighbour-list wrappers: tagged default, both modes, n_real padding, a stable sender shape
    delta_all = R[:, None, :] - R[None, :, :]
    delta_all -= onp.round(delta_all)
    pair_dist = onp.sqrt(onp.sum((delta_all @ onp.asarray(box).T) ** 2, -1))
    pair_senders, pair_receivers = onp.nonzero((pair_dist < CUTOFF) & ~onp.eye(N, dtype=bool))
    pair_idx = jnp.asarray(onp.stack([pair_senders, pair_receivers]), jnp.int32)
    E_pair_ref = float(domain_energy(jnp.asarray(R), pair_idx, 0, N))
    F_pair_ref = -onp.asarray(jax.grad(lambda r: domain_energy(r, pair_idx, 0, N))(jnp.asarray(R)))
    wrapper_ok = True
    details = []
    for mode in (None, "sender", "replicated"):
        energy_w = replicate_data.make_sharded_energy_with_nbrs(domain_energy, 8, N // 8, neighbor_sharding=mode)
        force_w = replicate_data.make_sharded_force_with_nbrs(domain_energy, 8, N // 8, neighbor_sharding=mode,
                                                              use_reduce_scatter=True, max_edges_per_domain=16384)
        e_w = float(energy_w(jnp.asarray(R), pair_idx, N))
        f_w = onp.asarray(force_w(jnp.asarray(R), pair_idx))
        f_again = onp.asarray(force_w(jnp.asarray(R), pair_idx, N))
        wrapper_ok &= abs(e_w - E_pair_ref) < 1e-10 * abs(E_pair_ref) and onp.max(onp.abs(f_w - F_pair_ref)) < 1e-9 \
            and onp.max(onp.abs(f_again - F_pair_ref)) < 1e-9
        details.append(f"{mode}: e_err={abs(e_w - E_pair_ref):.1e} f_err={onp.max(onp.abs(f_w - F_pair_ref)):.1e}")
    try:
        replicate_data.make_sharded_energy_with_nbrs(lambda R_all, nbrs, start, n: 0.0, 8, N // 8)
        untagged_raises = False
    except ValueError:
        untagged_raises = True
    try:
        replicate_data.shard_neighbor_idx_by_sender(pair_idx, 8, N // 8, max_edges_per_domain=16)
        capacity_raises = False
    except ValueError:
        capacity_raises = True
    check("with_nbrs wrappers", wrapper_ok and untagged_raises and capacity_raises,
          "; ".join(details) + f" untagged_raises={untagged_raises} capacity_raises={capacity_raises}")

    # padded atoms beyond n_real contribute nothing and receive zero force, in both modes
    n_pad = 8
    N_pad = N + n_pad
    R_pad = jnp.asarray(onp.concatenate([R, (R[:n_pad] + 1e-3) % 1.0]))
    pad_edges = onp.stack([onp.r_[N + onp.arange(n_pad), onp.arange(n_pad)],
                           onp.r_[onp.arange(n_pad), N + onp.arange(n_pad)]])
    pair_idx_pad = jnp.asarray(onp.concatenate([onp.asarray(pair_idx), pad_edges], 1), jnp.int32)
    pad_ok = True
    details = []
    for mode in ("sender", "replicated"):
        energy_p = replicate_data.make_sharded_energy_with_nbrs(domain_energy, 8, N_pad // 8, neighbor_sharding=mode)
        force_p = replicate_data.make_sharded_force_with_nbrs(domain_energy, 8, N_pad // 8, neighbor_sharding=mode)
        e_p = float(energy_p(R_pad, pair_idx_pad, N))
        f_p = onp.asarray(force_p(R_pad, pair_idx_pad, N))
        e_unmasked = float(energy_p(R_pad, pair_idx_pad))
        pad_ok &= abs(e_p - E_pair_ref) < 1e-10 * abs(E_pair_ref) and onp.max(onp.abs(f_p[:N] - F_pair_ref)) < 1e-9 \
            and onp.max(onp.abs(f_p[N:])) == 0.0 and abs(e_unmasked - E_pair_ref) > 1e-3
        details.append(f"{mode}: e_err={abs(e_p - E_pair_ref):.1e} f_err={onp.max(onp.abs(f_p[:N] - F_pair_ref)):.1e} "
                       f"pad_force={onp.max(onp.abs(f_p[N:])):.1e} unmasked_diff={abs(e_unmasked - E_pair_ref):.2f}")
    check("with_nbrs n_real padding", pad_ok, "; ".join(details))

    # EAM neighbour energy: replicated default, sender refused, dense reference
    charge_fn = lambda dist: jnp.where(dist < CUTOFF, jnp.exp(-2.0 * dist), 0.0)
    embed_fn = lambda rho: -jnp.sqrt(rho + 1e-12)
    eam_dense = replicate_data.domain_energy_from_eam(charge_fn, embed_fn, pair, displacement)
    eam_nb = replicate_data.domain_energy_from_eam_with_neighbors(charge_fn, embed_fn, pair, displacement)
    E_eam_ref = float(eam_dense(jnp.asarray(R), 0, N))
    F_eam_ref = -onp.asarray(jax.grad(lambda r: eam_dense(r, 0, N))(jnp.asarray(R)))
    e_eam = float(replicate_data.make_sharded_energy_with_nbrs(eam_nb, 8, N // 8)(jnp.asarray(R), pair_idx))
    f_eam = onp.asarray(replicate_data.make_sharded_force_with_nbrs(eam_nb, 8, N // 8, use_reduce_scatter=True)(
        jnp.asarray(R), pair_idx))
    try:
        replicate_data.make_sharded_energy_with_nbrs(eam_nb, 8, N // 8, neighbor_sharding="sender")
        eam_sender_raises = False
    except ValueError:
        eam_sender_raises = True
    check("eam with_nbrs", abs(e_eam - E_eam_ref) < 1e-10 * abs(E_eam_ref)
          and onp.max(onp.abs(f_eam - F_eam_ref)) < 1e-9 and eam_sender_raises,
          f"e_err={abs(e_eam - E_eam_ref):.1e} f_err={onp.max(onp.abs(f_eam - F_eam_ref)):.1e} "
          f"sender_raises={eam_sender_raises}")

    # dense replicated-data wrappers with both folds; jax_md forces are real-space
    dom_pair = replicate_data.domain_energy_from_pair(pair_fn, displacement)
    cfg = replicate_data.create_config(8, N // 8)
    dense_ok = abs(float(replicate_data.make_sharded_energy(dom_pair, cfg)(jnp.asarray(R))) - E_ref) < 1e-10 * abs(E_ref)
    for reduce_scatter in (False, True):
        f_d = onp.asarray(replicate_data.make_sharded_force(dom_pair, cfg, use_reduce_scatter=reduce_scatter)(
            jnp.asarray(R)))
        e_f, f_f = replicate_data.make_sharded_energy_force(dom_pair, cfg, use_reduce_scatter=reduce_scatter)(
            jnp.asarray(R).reshape(8, N // 8, 3))
        dense_ok &= onp.max(onp.abs(f_d - F_ref / L)) < 1e-9 and abs(float(e_f) - E_ref) < 1e-10 * abs(E_ref) \
            and onp.max(onp.abs(onp.asarray(f_f) - F_ref / L)) < 1e-9
    check("dense wrappers", dense_ok)

    # input guards of the replicated wrappers
    try:
        replicate_data.create_config(8, N // 8, mesh=Mesh(jax.devices()[:4], axis_names=("i",)))
        mesh_raises = False
    except ValueError:
        mesh_raises = True
    by_receiver = replicate_data.shard_neighbor_idx_by_sender(pair_idx[::-1], 8, N // 8)[:, ::-1, :]
    try:
        replicate_data.make_sharded_energy_with_nbrs(domain_energy, 8, N // 8)(jnp.asarray(R), by_receiver)
        partition_raises = False
    except ValueError:
        partition_raises = True
    tags_ok = (replicate_data.check_neighbor_sharding(partial(eam_nb), None) == "replicated"
               and replicate_data.check_neighbor_sharding(jax.tree_util.Partial(domain_energy), None) == "sender")
    try:
        replicate_data.check_neighbor_sharding(partial(eam_nb), "sender")
        tags_ok = False
    except ValueError:
        pass
    padded_hi = jnp.asarray([[0, 1, 2], [1, 0, N]], jnp.int32)
    try:
        replicate_data.validate_symmetric_neighbor_list(padded_hi, n_atoms=N)
        pad_hi_ok = True
    except ValueError:
        pad_hi_ok = False
    try:
        replicate_data.validate_symmetric_neighbor_list(padded_hi)
        pad_hi_ok = False
    except ValueError:
        pass
    n_real_ok = (not replicate_data.energy_fn_accepts_n_real_arg(lambda R_all, nbrs, start, n, scale=1.0: 0.0)
                 and replicate_data.energy_fn_accepts_n_real_arg(lambda R_all, nbrs, start, n, n_real=None: 0.0)
                 and replicate_data.energy_fn_accepts_n_real_arg(lambda R_all, nbrs, start, n, **kw: 0.0))
    n_real_ok &= not replicate_data.energy_fn_accepts_n_real_arg(partial(domain_energy, n_real=N))
    plain_energy = lambda R_all, nbrs, start, n: 0.0
    try:
        replicate_data.make_sharded_energy_with_nbrs(plain_energy, 8, N // 8, n_real=N, neighbor_sharding="sender")
        n_real_ok = False
    except ValueError:
        pass
    energy_no_n_real = replicate_data.make_sharded_energy_with_nbrs(plain_energy, 8, N // 8, neighbor_sharding="sender",
                                                                    validate_neighbor_symmetry=False)
    try:
        energy_no_n_real(jnp.asarray(R), pair_idx, N)
        n_real_ok = False
    except ValueError:
        pass
    check("with_nbrs guards", mesh_raises and partition_raises and tags_ok and pad_hi_ok and n_real_ok,
          f"mesh_raises={mesh_raises} partition_raises={partition_raises} tags_ok={tags_ok} "
          f"pad_hi_ok={pad_hi_ok} n_real_ok={n_real_ok}")

    # the neighbour energies avoid the vmapped metric form that XLA CPU miscompiles in float64 for
    # roughly 16385 to 40000 edges; pin an in-window list through the replicated wrapper
    window = (pair_senders < 600) & (pair_receivers < 600)
    window_idx = jnp.asarray(onp.stack([pair_senders[window], pair_receivers[window]]), jnp.int32)
    e_window = float(replicate_data.make_sharded_energy_with_nbrs(
        domain_energy, 8, N // 8, neighbor_sharding="replicated")(jnp.asarray(R), window_idx))
    e_window_ref = float(domain_energy(jnp.asarray(R), window_idx, 0, N))
    check("metric window", 16384 < window_idx.shape[1] < 40000
          and abs(e_window - e_window_ref) < 1e-10 * abs(e_window_ref),
          f"edges={window_idx.shape[1]} e_err={abs(e_window - e_window_ref):.1e}")
    R16 = jnp.asarray(R[:16])
    clean = jnp.asarray([[0, 1], [1, 0]], jnp.int32).T
    padded = jnp.asarray([[0, 1, 2], [1, 0, -1]], jnp.int32)
    e_clean = float(domain_energy(R16, clean, 0, 16))
    e_padded = float(domain_energy(R16, padded, 0, 16))
    check("mixed pad edge", abs(e_clean - e_padded) < 1e-14,
          f"{e_clean:.12f} vs {e_padded:.12f}")

    print("ALL PASS" if not failures else f"FAILURES: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    run_validation()
