"""Per-layer feature exchange for LAMMPS-JAX exports.

`Comm.forward_comm` lowers to the ``lammps_jax.forward_comm`` FFI custom call, filling
ghost rows from owner ranks; a threaded float32 token keeps exchanges ordered under XLA.
"""

import math
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

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
    _leaves, _treedef, _n_rows, leaf_widths = flatten_features(features)
    return int(sum(leaf_widths))


def forward_comm(features: Any) -> Any:
    """Identity stand-in for `Comm.forward_comm` without domain decomposition.

    Every row is owned, so forward communication changes nothing.
    """
    flatten_features(features)
    return features


def exchange_call(target: str, matrix: jax.Array, token: jax.Array):
    """Emit one comm custom call: (matrix, token) -> (matrix, token).

    Inputs alias outputs; when XLA must copy a still-live buffer, the handler
    falls back to a full identity copy.
    """
    call = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
            jax.ShapeDtypeStruct(token.shape, token.dtype),
        ),
        input_output_aliases={0: 0, 1: 1},
    )
    matrix_out, token_out = call(matrix, token)
    return matrix_out, token_out


@jax.custom_vjp
def forward_exchange(matrix: jax.Array, token: jax.Array):
    return exchange_call(FORWARD_TARGET, matrix, token)


def forward_exchange_fwd(matrix: jax.Array, token: jax.Array):
    return forward_exchange(matrix, token), None


def forward_exchange_bwd(_residuals, cotangents):
    # Reverse comm: sum ghost cotangents into owner rows and zero the ghosts.
    # The threaded token cotangent keeps backward sites in mirror order.
    matrix_cotangent, token_cotangent = cotangents
    return exchange_call(REVERSE_TARGET, matrix_cotangent, token_cotangent)


forward_exchange.defvjp(forward_exchange_fwd, forward_exchange_bwd)


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
        self._token: jax.Array | None = None

    def forward_comm(self, features: Any) -> Any:
        """Exchange one pytree of per-atom features, filling ghost rows from owner ranks.

        Differentiable: the VJP is a reverse exchange of the cotangent.
        """
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
        if not self.enabled:
            return features

        # The FFI handler exchanges a single [n_rows, width] float32 matrix.
        columns = [
            jnp.reshape(leaf, (n_rows, leaf_width)).astype(jnp.float32)
            for leaf, leaf_width in zip(leaves, leaf_widths)
        ]
        matrix = columns[0] if len(columns) == 1 else jnp.concatenate(columns, axis=1)
        if self._token is None:
            self._token = jnp.zeros((), dtype=jnp.float32)
        matrix, self._token = forward_exchange(matrix, self._token)

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
