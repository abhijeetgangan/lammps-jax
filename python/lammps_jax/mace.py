"""MACE-JAX adapter for the exported-graph ABI in the ghost and comm schemes.

`edge_index` swaps graph roles: neighbors send, row owners receive, so owned
atoms aggregate complete neighborhoods.
"""

import copy
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp


def collapse_skip(block: Any, elements: Sequence[int], num_elements: int) -> Any:
    """Copy of an interaction block whose skip connection is one matmul per element.

    The skip tensor product is linear in node_feats for a fixed one-hot element,
    so probing it once per exported element is exact and skips the dense
    contraction over all num_elements channels. Blocks whose skip is not that
    tensor product (a plain linear layer) are returned unchanged. Rows whose
    element is not in elements come out NaN rather than silently wrong.
    """
    from flax import nnx
    from mace_jax.adapters.cuequivariance import FullyConnectedTensorProduct

    if not isinstance(getattr(block, "skip_tp", None), FullyConnectedTensorProduct):
        return block

    def skip(node_feats, node_attrs):
        width = node_feats.shape[1]
        with jax.ensure_compile_time_eval():
            eye = jnp.eye(width, dtype=node_feats.dtype)
            mats = [block.skip_tp(eye, jax.nn.one_hot(jnp.full((width,), t), num_elements,
                                                      dtype=node_feats.dtype))
                    for t in elements]
        species = jnp.argmax(node_attrs, axis=1)
        out = jnp.zeros((node_feats.shape[0], mats[0].shape[1]), node_feats.dtype)
        covered = jnp.zeros((node_feats.shape[0],), bool)
        for t, mat in zip(elements, mats):
            product = jnp.matmul(node_feats, mat, precision=jax.lax.Precision.HIGHEST)
            out = out + jnp.where((species == t)[:, None], product, 0.0)
            covered = covered | (species == t)
        return jnp.where(covered[:, None], out, jnp.nan)

    patched = copy.copy(block)
    patched.skip_tp = nnx.static(skip)
    return patched


def make_mace_energy(
    *,
    config: dict[str, Any],
    model: Any,
    communicating: bool = False,
    owned_rows: int | None = None,
    elements: Sequence[int] | None = None,
) -> Callable[..., Any]:
    """Build a per-atom MACE energy callable with the exported-model signature.

    The comm form runs `comm.forward_comm(node_feats)` before every interaction
    after the first; the ghost form omits the `comm` argument. owned_rows
    truncates the product basis to the leading owned rows, whose ghosts the
    exchange refreshes anyway. elements lists every model element index the
    species input can carry; the interaction skip connections then run per
    element instead of over every one-hot channel, and any other element
    yields NaN energies.
    """
    if owned_rows is not None and not communicating:
        raise ValueError("owned-row truncation needs a communicating export: "
                         "ghost features must arrive through the exchange")
    r_max = jnp.float32(config["r_max"])
    num_elements = int(config["num_elements"])
    interactions = list(model.interactions)
    if elements is not None:
        elements = tuple(sorted({int(t) for t in elements}))
        if not elements or not all(0 <= t < num_elements for t in elements):
            raise ValueError(f"elements must index the model's {num_elements} elements; "
                             f"got {elements}")
        interactions = [collapse_skip(block, elements, num_elements) for block in interactions]

    def node_energies(positions, species, graph, comm=None):
        n_atoms = positions.shape[0]
        # Row owners aggregate; neighbors send. mace_jax and the fused kernels index
        # these themselves, so padded slots stay in bounds at row 0.
        centers = jnp.where(graph.edge_mask, graph.senders, 0)
        neighbors = jnp.where(graph.edge_mask, graph.receivers, 0)
        edge_index = jnp.stack([neighbors, centers], axis=0)

        vectors = positions[centers] - positions[neighbors]
        pad_vector = jnp.array([1.0, 0.0, 0.0], dtype=vectors.dtype) * r_max
        vectors = jnp.where(graph.edge_mask[:, None], vectors, pad_vector)
        lengths = jnp.linalg.norm(vectors, axis=-1, keepdims=True)

        safe_species = jnp.clip(species, 0, num_elements - 1)
        node_attrs = jax.nn.one_hot(safe_species, num_elements, dtype=vectors.dtype)
        node_attrs_index = safe_species.astype(jnp.int32)
        node_heads = jnp.zeros((n_atoms,), dtype=jnp.int32)
        arange = jnp.arange(n_atoms)

        node_e0 = model.atomic_energies_fn(node_attrs)[arange, node_heads]

        node_feats = model.node_embedding(node_attrs)
        edge_attrs = model.spherical_harmonics(vectors)
        edge_feats, cutoff = model.radial_embedding(
            lengths,
            node_attrs,
            edge_index,
            model._atomic_numbers,
            node_attrs_index=node_attrs_index,
        )
        node_energies_list = []
        node_feats_list = []
        for idx, (interaction, product) in enumerate(zip(interactions, model.products)):
            if comm is not None and idx > 0:
                # Refresh ghosts from owner ranks; mace_jax's ML-IAP exchange point.
                node_feats = comm.forward_comm(node_feats)
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                n_real=None,
                first_layer=(idx == 0),
            )
            if owned_rows is not None:
                truncated = product(
                    node_feats=node_feats[:owned_rows],
                    sc=None if sc is None else sc[:owned_rows],
                    node_attrs=node_attrs[:owned_rows],
                    node_attrs_index=node_attrs_index[:owned_rows],
                )
                node_feats = jnp.zeros(
                    (n_atoms,) + truncated.shape[1:], truncated.dtype
                ).at[:owned_rows].set(truncated)
            else:
                node_feats = product(
                    node_feats=node_feats,
                    sc=sc,
                    node_attrs=node_attrs,
                    node_attrs_index=node_attrs_index,
                )
            node_feats_list.append(node_feats)

        for idx, readout in enumerate(model.readouts):
            feat_idx = -1 if len(model.readouts) == 1 else idx
            node_energies_list.append(
                readout(node_feats_list[feat_idx], node_heads)[arange, node_heads]
            )

        node_inter_es = jnp.sum(jnp.stack(node_energies_list, axis=0), axis=0)
        node_inter_es = model.scale_shift(node_inter_es, node_heads)
        return node_e0 + node_inter_es

    if communicating:
        # Positional comm: exporting without comm=True must fail, not skip.
        def comm_energy(positions, species, graph, comm):
            return node_energies(positions, species, graph, comm)

        return comm_energy

    def plain_energy(positions, species, graph):
        return node_energies(positions, species, graph)

    return plain_energy
