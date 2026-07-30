"""MACE-JAX adapter for the exported-graph ABI in the ghost and comm schemes.

`edge_index` swaps graph roles: neighbors send, row owners receive, so owned
atoms aggregate complete neighborhoods.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def make_mace_energy(
    *,
    config: dict[str, Any],
    model: Any,
    communicating: bool = False,
) -> Callable[..., Any]:
    """Build a per-atom MACE energy callable with the exported-model signature.

    The comm form runs `comm.forward_comm(node_feats)` before every interaction
    after the first; the ghost form omits the `comm` argument.
    """
    r_max = jnp.float32(config["r_max"])
    num_elements = int(config["num_elements"])

    def node_energies(positions, species, graph, comm=None):
        n_atoms = positions.shape[0]
        # Row owners aggregate; neighbors send.
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
        for idx, (interaction, product) in enumerate(
            zip(model.interactions, model.products)
        ):
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
