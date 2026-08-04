"""Per-layer feature exchange for LAMMPS-JAX exports.

`Comm.forward_comm` lowers to the ``lammps_jax.forward_comm`` FFI custom call, filling
ghost rows from owner ranks; a threaded float32 token keeps exchanges ordered under XLA.
Exchanges are linear primitives whose transposes swap forward and reverse, so every
differentiation mode composes.
"""

import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import jax
import jax.extend.core
import jax.numpy as jnp
from jax.interpreters import ad, batching, mlir

FORWARD_TARGET = "lammps_jax.forward_comm"
REVERSE_TARGET = "lammps_jax.reverse_comm"

# The exporter allowlists these via jax.export.DisabledSafetyCheck.custom_call.
CUSTOM_CALL_TARGETS = (FORWARD_TARGET, REVERSE_TARGET)


def flatten_features(features: Any) -> tuple[list[Any], Any, int, list[int]]:
    """Flatten and validate an exchange payload into (leaves, treedef, n_rows, leaf_widths).

    Leaves must be float arrays sharing an atom-leading axis.
    """
    leaves, treedef = jax.tree.flatten(features)
    if not leaves:
        raise ValueError("comm features must contain at least one array leaf")
    leading_sizes = []
    for index, leaf in enumerate(leaves):
        dtype = jnp.result_type(leaf)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError(
                "comm features must have a floating dtype so exchanges are "
                f"differentiable; leaf {index} has dtype {dtype}"
            )
        shape = jnp.shape(leaf)
        if len(shape) == 0:
            raise ValueError(
                "comm features must be shaped [n_atoms, ...] with an "
                f"atom-leading axis; leaf {index} is a scalar"
            )
        leading_sizes.append(shape[0])
    if len(set(leading_sizes)) > 1:
        raise ValueError(
            "comm feature leaves disagree on the atom-leading dimension; "
            f"got leading sizes {leading_sizes}"
        )
    leaf_widths = [math.prod(jnp.shape(leaf)[1:]) for leaf in leaves]
    return leaves, treedef, leading_sizes[0], leaf_widths


def comm_width(features: Any) -> int:
    """Total exchanged width of a feature pytree.

    Float32 columns packed per atom row: sum over leaves of trailing-dimension products.
    """
    leaf_widths = flatten_features(features)[3]
    return int(sum(leaf_widths))


def forward_comm(features: Any) -> Any:
    """Identity stand-in for `Comm.forward_comm` without domain decomposition.

    Every row is owned, so forward communication changes nothing.
    """
    flatten_features(features)
    return features


def exchange_jvp(primitive, primals, tangents):
    # The exchange is linear: tangents travel through the same exchange.
    matrix_dot, token_dot = tangents
    outs = primitive.bind(*primals)
    if type(matrix_dot) is ad.Zero and type(token_dot) is ad.Zero:
        return outs, tuple(ad.Zero(out.aval.to_tangent_aval()) for out in outs)
    tangent_outs = primitive.bind(
        ad.instantiate_zeros(matrix_dot), ad.instantiate_zeros(token_dot)
    )
    return outs, tangent_outs


def exchange_transpose(adjoint, cotangents, matrix, token):
    # The token cotangent keeps transposed sites in mirror order.
    matrix_cotangent, token_cotangent = map(ad.instantiate_zeros, cotangents)
    out_matrix, out_token = adjoint.bind(matrix_cotangent, token_cotangent)
    return (
        out_matrix if ad.is_undefined_primal(matrix) else None,
        out_token if ad.is_undefined_primal(token) else None,
    )


def exchange_batch(primitive, batched_args, batch_dims):
    # The handler forwards whole rows: the batch folds into the width.
    matrix, token = batched_args
    matrix_dim, token_dim = batch_dims
    if token_dim is not None:
        token = jax.lax.index_in_dim(token, 0, token_dim, keepdims=False)
    if matrix_dim is None:
        return primitive.bind(matrix, token), (None, None)
    folded = jnp.moveaxis(matrix, matrix_dim, 2)
    rows, width, batch = folded.shape
    out_matrix, out_token = primitive.bind(
        jnp.reshape(folded, (rows, width * batch)), token
    )
    out_matrix = jnp.moveaxis(
        jnp.reshape(out_matrix, (rows, width, batch)), 2, matrix_dim
    )
    return (out_matrix, out_token), (matrix_dim, None)


def make_exchange_primitive(name: str, target: str):
    """Define one comm custom call primitive: (matrix, token) -> (matrix, token).

    Inputs alias outputs; when XLA must copy a still-live buffer, the handler
    falls back to a full identity copy.
    """
    primitive = jax.extend.core.Primitive(name)
    primitive.multiple_results = True
    primitive.def_abstract_eval(lambda matrix, token: (matrix, token))
    primitive.def_impl(jax.jit(lambda matrix, token: primitive.bind(matrix, token)))
    mlir.register_lowering(
        primitive, jax.ffi.ffi_lowering(target, operand_output_aliases={0: 0, 1: 1})
    )
    ad.primitive_jvps[primitive] = partial(exchange_jvp, primitive)
    batching.primitive_batchers[primitive] = partial(exchange_batch, primitive)
    return primitive


forward_exchange_p = make_exchange_primitive("forward_exchange", FORWARD_TARGET)
reverse_exchange_p = make_exchange_primitive("reverse_exchange", REVERSE_TARGET)
ad.primitive_transposes[forward_exchange_p] = partial(
    exchange_transpose, reverse_exchange_p
)
ad.primitive_transposes[reverse_exchange_p] = partial(
    exchange_transpose, forward_exchange_p
)


def forward_exchange(matrix: jax.Array, token: jax.Array):
    """Fill ghost rows of a [n_rows, width] float32 matrix from owner ranks."""
    return forward_exchange_p.bind(matrix, token)


def reverse_exchange(matrix: jax.Array, token: jax.Array):
    """Sum ghost rows into owner rows and zero the ghosts."""
    return reverse_exchange_p.bind(matrix, token)


class Comm:
    """Trace-local exchange state: token chain, width record, schedule check.

    Use one instance per trace; with ``enabled=False`` exchanges are identities
    but widths are still recorded and checked against ``expected_widths``.
    """

    def __init__(
        self, enabled: bool = True, expected_widths: Sequence[int] | None = None
    ):
        self.enabled = bool(enabled)
        self.expected_widths = (
            None if expected_widths is None else tuple(int(w) for w in expected_widths)
        )
        self.widths: list[int] = []
        self.kinds: list[str] = []
        self.token: jax.Array | None = None

    def forward_comm(self, features: Any) -> Any:
        """Exchange one pytree of per-atom features, filling ghost rows from owner ranks.

        Linear in the features: differentiating in any mode emits the adjoint
        reverse exchange for cotangents and a forward exchange for tangents.
        """
        return self.exchange(features, forward_exchange, "forward")

    def reverse_comm(self, features: Any) -> Any:
        """Sum ghost-row features into their owner rows and zero the ghosts.

        The adjoint of forward_comm, for models that accumulate partial sums
        on ghost rows.
        """
        return self.exchange(features, reverse_exchange, "reverse")

    def exchange(self, features: Any, exchange_fn: Callable[..., Any],
                 kind: str) -> Any:
        leaves, treedef, n_rows, leaf_widths = flatten_features(features)
        width = int(sum(leaf_widths))
        site = len(self.widths)
        if self.expected_widths is not None:
            if site >= len(self.expected_widths):
                raise ValueError(
                    f"comm exchange {site} falls outside the declared schedule "
                    f"of {len(self.expected_widths)} site(s)"
                )
            expected = self.expected_widths[site]
            if width != expected:
                raise ValueError(
                    f"comm exchange {site} has width {width}, expected {expected}"
                )
        self.widths.append(width)
        self.kinds.append(kind)
        if not self.enabled:
            return features

        # The FFI handler exchanges a single [n_rows, width] float32 matrix.
        columns = [
            jnp.reshape(leaf, (n_rows, leaf_width)).astype(jnp.float32)
            for leaf, leaf_width in zip(leaves, leaf_widths)
        ]
        matrix = columns[0] if len(columns) == 1 else jnp.concatenate(columns, axis=1)
        if self.token is None:
            self.token = jnp.zeros((), dtype=jnp.float32)
        matrix, self.token = exchange_fn(matrix, self.token)

        pieces = []
        offset = 0
        for leaf, leaf_width in zip(leaves, leaf_widths):
            block = jax.lax.slice_in_dim(matrix, offset, offset + leaf_width, axis=1)
            offset += leaf_width
            pieces.append(
                jnp.reshape(block, jnp.shape(leaf)).astype(jnp.result_type(leaf))
            )
        return jax.tree.unflatten(treedef, pieces)

    def validate(self) -> None:
        """Check that the trace visited a prefix of the declared schedule in order.

        A bundle's programs share one schedule; a program with fewer exchange
        sites than the fused one visits a leading subset.
        """
        if self.expected_widths is None:
            return
        observed = tuple(self.widths)
        if observed != self.expected_widths[: len(observed)]:
            raise ValueError(
                "comm exchange schedule mismatch: observed widths "
                f"{list(self.widths)}, expected a prefix of "
                f"{list(self.expected_widths)}"
            )
