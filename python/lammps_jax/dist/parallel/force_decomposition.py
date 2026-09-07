"""Force-decomposition parallelism for isotropic pair potentials on a 2-D mesh.

P = q * q devices tile the (N, N) pair matrix into (N / q, N / q) blocks. Each
device all-gathers its row and column position blocks, evaluates the block
energy with the 0.5 weight of Plimpton's F1 scheme, and folds the row and
column gradients onto its owned N / P atoms with tiled ``lax.psum_scatter``.
Communication per device is O(N / sqrt(P)) versus O(N) for replicated data;
the dense block costs O((N / sqrt(P))^2) memory, which limits it to small systems.

References:
    S. Plimpton, "Fast Parallel Algorithms for Short-Range Molecular
    Dynamics", J. Comput. Phys. 117, 1-19 (1995),
    doi:10.1006/jcph.1995.1039.
    A. P. Thompson et al., "LAMMPS - a flexible simulation tool for
    particle-based materials modeling at the atomic, meso, and continuum
    scales", Comput. Phys. Commun. 271, 108171 (2022),
    doi:10.1016/j.cpc.2021.108171.
"""

import math
from functools import partial
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as onp
from jax import lax, shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax_md import space

Array = jnp.ndarray


class ForceConfig(NamedTuple):
    """Mesh and block sizes for the force-decomposition wrapper.

    Attributes:
        mesh: (q, q) device mesh with axes row and col.
        q: Side of the mesh; P = q * q devices.
        n_sub: Atoms owned per device; N = P * n_sub.
    """

    mesh: Mesh
    q: int
    n_sub: int


def create_config(n_atoms: int, n_devices: int,
                  mesh: Optional[Mesh] = None) -> ForceConfig:
    """Builds a ForceConfig; mesh defaults to the first P local devices as (q, q).

    Raises:
        ValueError: P is not a perfect square or N % P != 0.
    """
    q = math.isqrt(n_devices)
    if q * q != n_devices:
        raise ValueError('n_devices must be a perfect square for a square 2-D mesh.')
    if n_atoms % n_devices != 0:
        raise ValueError('n_atoms must be divisible by n_devices.')
    if mesh is None:
        devices = onp.asarray(jax.devices()[:n_devices]).reshape(q, q)
        mesh = Mesh(devices, axis_names=('row', 'col'))
    return ForceConfig(mesh=mesh, q=q, n_sub=n_atoms // n_devices)


def block_global_ids(q: int, n_sub: int) -> tuple[Array, Array]:
    """Global atom ids of this device's row block and column block, each (N // q,)."""
    row = lax.axis_index('row')
    col = lax.axis_index('col')
    block_index = jnp.arange(q, dtype=jnp.int32)
    atom_index = jnp.arange(n_sub, dtype=jnp.int32)
    row_ids = ((row * q + block_index)[:, None] * n_sub + atom_index[None, :]).reshape(-1)
    col_ids = ((block_index * q + col)[:, None] * n_sub + atom_index[None, :]).reshape(-1)
    return (row_ids, col_ids)


def block_energy_from_pair(pair_energy_fn: Callable,
                           displacement_fn: Callable) -> Callable:
    """Builds a dense-distance block energy from an isotropic pair potential.

    Args:
        pair_energy_fn: Vectorized ``dr -> energy`` over a distance array.

    Returns:
        ``(row_block (N // q, 3), col_block (N // q, 3), self_mask (N // q, N // q)) -> scalar``
        with masked entries zeroed and each pair weighted 0.5.
    """
    compute_distances = lambda row_block, col_block: space.distance(jax.vmap(
        lambda r_i: jax.vmap(lambda r_j: displacement_fn(r_i, r_j))(col_block))(row_block))

    def block_energy(row_block, col_block, self_mask):
        dr = compute_distances(row_block, col_block)
        safe_dr = jnp.where(self_mask, jnp.ones_like(dr), dr)
        pair_energies = jnp.where(self_mask, 0.0, pair_energy_fn(safe_dr))
        return 0.5 * jnp.sum(pair_energies)

    return block_energy


def make_sharded_energy_force(pair_energy_fn: Callable, displacement_fn: Callable,
                              config: ForceConfig) -> Callable:
    """Force decomposition ``positions (N, 3) -> (energy, forces (N, 3))``.

    Args:
        pair_energy_fn: Vectorized ``dr -> energy`` over a distance array.
        displacement_fn: jax-md displacement on fractional coordinates; forces are real-space.
        config: Mesh and block sizes; device (r, c) owns atoms (r * q + c) * n_sub + k.
    """
    mesh = config.mesh
    q = config.q
    n_sub = config.n_sub
    sharding = NamedSharding(mesh, P('row', 'col'))
    block_energy = block_energy_from_pair(pair_energy_fn, displacement_fn)

    @partial(shard_map, mesh=mesh, in_specs=(P('row', 'col'),),
             out_specs=(P(), P('row', 'col')), check_vma=False)
    def sharded_energy_force(R_sub):
        row_block = lax.all_gather(R_sub, 'col', axis=1, tiled=True).reshape(-1, 3)
        col_block = lax.all_gather(R_sub, 'row', axis=0, tiled=True).reshape(-1, 3)
        row_ids, col_ids = block_global_ids(q, n_sub)
        self_mask = row_ids[:, None] == col_ids[None, :]
        E_block, (grad_row, grad_col) = jax.value_and_grad(
            block_energy, argnums=(0, 1))(row_block, col_block, self_mask)
        fold_row = lax.psum_scatter(grad_row.reshape(1, q, n_sub, 3), 'col',
                                    scatter_dimension=1, tiled=True)
        fold_col = lax.psum_scatter(grad_col.reshape(q, 1, n_sub, 3), 'row',
                                    scatter_dimension=0, tiled=True)
        E_total = lax.psum(E_block, ('row', 'col'))
        F_sub = -(fold_row + fold_col).reshape(1, 1, n_sub, 3)
        return (E_total, F_sub)

    @jax.jit
    def wrapped(R):
        R_on_mesh = jax.device_put(R.reshape(q, q, n_sub, 3), sharding)
        E, F = sharded_energy_force(R_on_mesh)
        return (E, F.reshape(-1, 3))

    return wrapped
