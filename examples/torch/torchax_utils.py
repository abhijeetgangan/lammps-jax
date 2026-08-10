"""torchax interop shared by the torch export scripts; importing also
sets the jax flags that embed baked weights as export constants."""

import copy

import numpy as np
import torch
import torchax
from ase.build import bulk
from torchax import interop

import jax
import jax.numpy as jnp

jax.config.update("jax_use_simplified_jaxpr_constants", True)
jax.config.update("jax_embedded_constants_max_bytes", 1 << 30)


def to_jax_fn(module):
    """Bake a pristine copy of a torch module into a jax callable."""
    env = torchax.default_env()
    module = copy.deepcopy(module)
    with env:
        module = module.to("jax")
    for sub in module.modules():
        # fx-generated blocks keep constants as plain attributes and python
        # lists of Parameters, which .to() misses; convert them in place.
        for name, value in list(vars(sub).items()):
            if isinstance(value, torch.Tensor) and not isinstance(
                    value, torchax.tensor.Tensor):
                setattr(sub, name, env.to_xla(value))
            elif isinstance(value, (list, tuple)) and value and all(
                    isinstance(item, torch.Tensor) for item in value):
                moved = [item if isinstance(item, torchax.tensor.Tensor)
                         else env.to_xla(item) for item in value]
                setattr(sub, name, type(value)(moved))
    return interop.jax_view(module.forward)


def rattled_crystal(r_max, element="Al", seed=3):
    """A rattled crystal cluster and its directed within-cutoff edge list.

    The rattle keeps every environment generic; a pristine lattice hides
    errors behind symmetric force cancellations.
    """
    atoms = bulk(element, cubic=True) * (2, 2, 2)
    atoms.rattle(0.05, seed=seed)
    cluster = atoms.positions.astype(np.float32)
    delta = cluster[None] - cluster[:, None]
    within = (np.sqrt((delta ** 2).sum(-1)) < r_max) & ~np.eye(
        len(cluster), dtype=bool)
    centers, neighbors = np.nonzero(within)
    return cluster, centers, neighbors


def gate_fn(torch_gate):
    """e3nn Gate as a jax callable; the split is by hand because torchax
    drops the fx Extract's narrow().copy_() view writes, returning zeros."""
    dims = [ir.dim for ir in torch_gate.sc.irreps_outs]
    flat = [i for inst in torch_gate.sc.cut.instructions for i in inst]
    assert flat == sorted(flat), "gate sortcut is not a contiguous split"
    act_scalars = to_jax_fn(torch_gate.act_scalars)
    act_gates = to_jax_fn(torch_gate.act_gates)
    mul = to_jax_fn(torch_gate.mul)

    def gate(x):
        scalars, gates, gated = jnp.split(
            x, [dims[0], dims[0] + dims[1]], axis=-1)
        scalars = act_scalars(scalars)
        if dims[1] == 0:
            return scalars
        return jnp.concatenate([scalars, mul(gated, act_gates(gates))],
                               axis=-1)

    return gate


def species_linear(tp_module, attr_rows):
    """Probe a species-attribute FullyConnectedTensorProduct into one
    linear per species (notes/torch-import/xla-transpose-emitter-issue.md)."""
    fn = to_jax_fn(tp_module)
    dim = tp_module.irreps_in1.dim
    eye = jnp.eye(dim, dtype=jnp.float32)
    probe = jnp.asarray(np.random.default_rng(5).normal(size=(5, dim)),
                        jnp.float32)
    mats = []
    with jax.default_matmul_precision("highest"):
        for row in attr_rows:
            attrs = jnp.tile(row[None], (dim, 1))
            mats.append(fn(eye, attrs))
            ref = fn(probe, attrs[:5])
            error = float(jnp.max(jnp.abs(probe @ mats[-1] - ref)))
            assert error < 1e-6 * max(1.0, float(jnp.max(jnp.abs(ref)))), error
    return jnp.stack(mats)
