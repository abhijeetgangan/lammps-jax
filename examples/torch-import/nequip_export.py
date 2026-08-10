# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "lammps-jax",
#   "torchax==0.0.13",
#   "torch==2.8.0+cpu",
#   "jax[cuda12]==0.10.1",
#   "nequip==0.19.0",
# ]
#
# [tool.uv]
# index-strategy = "unsafe-best-match"
#
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
#
# [tool.uv.sources]
# lammps-jax = { path = "../..", editable = true }
# ///
"""Export NequIP checkpoints as lammps-jax comm bundles.

torchax converts the torch blocks to jax, forces come from jax.grad,
and the ghost exchange runs between interaction layers. For the
OpenEquivariance conv kernel use nequip_kernel_export.py.

Set OUTPUT and TYPE_Z, then run `uv run nequip_export.py`. The default
model downloads to ~/.nequip/model_cache. The Al deck
examples/in.mlip_al runs any bundle via -var bundle.
"""

from types import SimpleNamespace

import numpy as np
import torch

import jax
import jax.numpy as jnp
from ase.data import chemical_symbols
from nequip.model.saved_models.load_utils import load_saved_model

from lammps_jax.export import export_model
from torchax_utils import gate_fn, rattled_crystal, species_linear, to_jax_fn

OUTPUT = "nequip-oam.lammps-jax.json"
MODEL = "nequip.net:mir-group/NequIP-OAM-S:0.1"
TYPE_Z = [13]
MAX_ATOMS = 2048
MAX_OWNED = None
EDGES_PER_ATOM = 16
SKIP_CHECK = False


# Torch side: fetch the packaged model; nequip caches the download itself.
model = load_saved_model(MODEL).float().eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
r_max = float(model.metadata["r_max"])
type_names = model.metadata["type_names"].split(" ")
seq = model.model.func
blocks = dict(seq.named_children())
conv_names = sorted(n for n in blocks if n.endswith("_convnet"))
assert "pair_potential" not in blocks or type(
    blocks["pair_potential"]).__name__ == "ZBL", "unknown pair potential"
assert not seq.edge_norm._per_edge_type, "per-edge-type cutoffs not wired"
print(f"loaded: r_max {r_max}, {len(type_names)} types, "
      f"{len(conv_names)} conv layers")


# Convert: the interpreted callables.
sh_jax_fn = to_jax_fn(seq.spharm.sh)
cutoff_jax_fn = to_jax_fn(seq.bessel_encode.cutoff)
embed_jax_fn = to_jax_fn(seq.type_embed.embed_module)
readout_jax_fn = to_jax_fn(seq.per_atom_energy_readout.mlp_module)
zbl_jax_fn = (to_jax_fn(blocks["pair_potential"]._zbl)
              if "pair_potential" in blocks else None)

# Convert: buffers, per-type tables, and the ZBL constants; the bessel
# formula inlines from its buffer.
BESSEL_W = jnp.asarray(
    seq.bessel_encode.bessel_weights.detach().numpy().reshape(1, -1),
    jnp.float32)
BESSEL_FACTOR = float(seq.factor.factor)
SC_TYPES = jnp.asarray([type_names.index(chemical_symbols[z]) for z in TYPE_Z],
                       jnp.int32)
ATTR_ROWS = [embed_jax_fn(jnp.full((1,), t, jnp.int32))[0] for t in SC_TYPES]
SCALES = jnp.asarray(seq.per_type_energy_scale_shift.scales
                     .detach().numpy().astype(np.float32))
SHIFTS = jnp.asarray(seq.per_type_energy_scale_shift.shifts
                     .detach().numpy().astype(np.float32))
ZBL_Z = QQR2 = None
if zbl_jax_fn is not None:
    ZBL_Z = jnp.asarray(blocks["pair_potential"].atomic_numbers.detach()
                        .numpy().astype(np.float32))
    QQR2 = float(blocks["pair_potential"]._qqr2exesquare)

# Convert: conv layers. The sc tensor product collapses to a per-species
# linear since node_attrs is a pure species function.
LAYERS = []
for name in conv_names:
    conv_layer = blocks[name]
    block = conv_layer.conv
    assert not conv_layer.resnet, "resnet layers not wired"
    tp = to_jax_fn(block.tp_scatter.tp)

    def conv(h, edge_attrs, weights, centers, neighbors, tp=tp):
        edge_feats = tp(h[neighbors], edge_attrs, weights)
        return jnp.zeros((h.shape[0], edge_feats.shape[1]),
                         jnp.float32).at[centers].add(edge_feats)

    LAYERS.append({
        "linear_1": to_jax_fn(block.linear_1),
        "linear_2": to_jax_fn(block.linear_2),
        "edge_mlp": to_jax_fn(block.edge_mlp),
        "conv": conv,
        "gate": gate_fn(conv_layer.equivariant_nonlin),
        "sc_w": (species_linear(block.sc, ATTR_ROWS)
                 if block.sc is not None else None),
        "alpha": float(block.scatter_norm_factor),
        "first": block.is_first_layer,
    })

# Jax side: assemble the energy, exchanging features between layers; node
# work past each conv runs on the leading owned rows when owned is set.
def energy_fn(positions, species, graph, comm, owned=None):
    n_atoms = positions.shape[0]
    rows = owned or n_atoms
    centers = jnp.where(graph.edge_mask, graph.senders, 0)
    neighbors = jnp.where(graph.edge_mask, graph.receivers, 0)
    vectors = positions[neighbors] - positions[centers]
    pad_vector = jnp.array([1.0, 0.0, 0.0], jnp.float32) * r_max
    vectors = jnp.where(graph.edge_mask[:, None], vectors, pad_vector)
    lengths = jnp.linalg.norm(vectors, axis=-1, keepdims=True)

    normed = lengths / r_max
    bessel = jnp.sinc(normed * BESSEL_W) * BESSEL_W
    cut = cutoff_jax_fn(normed)
    edge_emb = bessel * cut * BESSEL_FACTOR
    edge_attrs = sh_jax_fn(vectors)

    node_attrs = embed_jax_fn(species)
    x = node_attrs
    for layer in LAYERS:
        sc = None
        if layer["sc_w"] is not None:
            sc = x[:rows] @ layer["sc_w"][0]
            for t in range(1, len(TYPE_Z)):
                sc = jnp.where((species[:rows] == SC_TYPES[t])[:, None],
                               x[:rows] @ layer["sc_w"][t], sc)
        h = layer["alpha"] * layer["linear_1"](x)
        if not layer["first"]:
            h = comm.forward_comm(h)
        weights = layer["edge_mlp"](edge_emb)
        weights = jnp.where(graph.edge_mask[:, None], weights, 0.0)
        h = layer["linear_2"](layer["conv"](h, edge_attrs, weights,
                                            centers, neighbors)[:rows])
        if sc is not None:
            h = h + sc
        x = layer["gate"](h)
        if owned is not None:
            # The exchange and the export contract expect full-length rows.
            x = jnp.zeros((n_atoms,) + x.shape[1:],
                          jnp.float32).at[:rows].set(x)

    energy = readout_jax_fn(x)[:, 0]
    energy = SHIFTS[species, 0] + SCALES[species, 0] * energy
    if zbl_jax_fn is not None:
        zbl_edge = zbl_jax_fn(ZBL_Z, lengths[:, 0], species,
                       jnp.stack([centers, neighbors]), QQR2)
        zbl_edge = jnp.where(graph.edge_mask, zbl_edge * cut[:, 0], 0.0)
        energy = energy + jnp.zeros((n_atoms,),
                                    jnp.float32).at[centers].add(zbl_edge)
    return energy


def verify(model):
    """Parity of the converted assembly against the torch model."""
    cluster, centers, neighbors = rattled_crystal(r_max)
    check_type = type_names.index(chemical_symbols[TYPE_Z[0]])
    reference = model({
        "pos": torch.tensor(cluster, requires_grad=True),
        "edge_index": torch.tensor(np.stack([centers, neighbors]),
                                   dtype=torch.long),
        "atom_types": torch.full((len(cluster),), check_type,
                                 dtype=torch.long),
        "num_nodes": torch.tensor([len(cluster)]),
    })
    reference_energy = float(reference["total_energy"].detach()[0])
    reference_forces = reference["forces"].detach().numpy()

    graph = SimpleNamespace(senders=jnp.asarray(centers, jnp.int32),
                            receivers=jnp.asarray(neighbors, jnp.int32),
                            edge_mask=jnp.ones(len(centers), bool))
    no_comm = SimpleNamespace(forward_comm=lambda features: features)
    species = jnp.full((len(cluster),), check_type, jnp.int32)
    with jax.default_matmul_precision("highest"):
        energy, grad = jax.value_and_grad(
            lambda p: jnp.sum(energy_fn(p, species, graph, no_comm)))(
                jnp.asarray(cluster))
    energy = float(energy)
    force_error = float(np.max(np.abs(-np.asarray(grad) - reference_forces)))
    print(f"parity vs torch: E {reference_energy:.7f} vs {energy:.7f} "
          f"(d {abs(reference_energy - energy):.2e}); max |dF| "
          f"{force_error:.2e} eV/A")
    assert abs(reference_energy - energy) < 1e-6 * abs(reference_energy), \
        "energy diverged from torch"
    assert force_error < 5e-5, "forces diverged from torch"


def export(output, custom_call_targets=()):
    # LAMMPS types map onto the model's element table.
    type_table = jnp.asarray(
        [type_names.index(chemical_symbols[z]) for z in TYPE_Z], jnp.int32)
    n_types = len(TYPE_Z)
    export_model(
        energy_fn=lambda positions, species, graph, comm: energy_fn(
            positions, type_table[jnp.clip(species, 0, n_types - 1)], graph,
            comm, MAX_OWNED),
        path=output,
        max_atoms=MAX_ATOMS,
        max_edges=MAX_ATOMS * EDGES_PER_ATOM,
        cutoff=r_max,
        unit_style="metal",
        comm=True,
        max_owned=MAX_OWNED,
        n_species=n_types,
        custom_call_targets=custom_call_targets,
    )
    print("exported", output)


if __name__ == "__main__":
    if not SKIP_CHECK:
        verify(model)
    export(OUTPUT)
