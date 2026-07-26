"""Export the example models for `pair_style jax/kk`: lj and eam subcommands.

`eam --mode comm` exports are float32 only; the plugin's comm staging buffers are f32.
"""

import argparse

import jax
import jax.numpy as jnp

from lammps_jax.eam import load_funcfl, load_setfl, make_eam_energy, make_setfl_energy
from lammps_jax.export import export_model


def lj_energy_terms(r_sq, mask, *, cutoff_sq, epsilon, sigma_sq):
    """Evaluate 12-6 energy per edge with autodiff-safe masking.

    Returns (valid, energy); energy on invalid edges must be masked by the caller.
    """
    dtype = r_sq.dtype
    valid = mask & (r_sq > jnp.asarray(1.0e-12, dtype)) & (r_sq < jnp.asarray(cutoff_sq, dtype))
    safe_r_sq = jnp.where(valid, r_sq, jnp.asarray(1.0, dtype))
    inv_r2 = jnp.asarray(sigma_sq, dtype) / safe_r_sq
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    energy = jnp.asarray(4.0 * epsilon, dtype) * (inv_r6 * inv_r6 - inv_r6)
    return valid, energy


def make_lj_energy(*, cutoff: float, epsilon: float, sigma: float, edge_energy_scale: float = 0.5):
    """Build an energy_fn with the exported-model signature.

    edge_energy_scale is 0.5 on full neighbor lists, 1.0 for newton-on edge-force exports.
    """
    params = dict(cutoff_sq=cutoff * cutoff, epsilon=epsilon, sigma_sq=sigma * sigma)

    def lj_energy(positions, species, graph):
        del species
        dtype = positions.dtype
        safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
        safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
        rij = positions[safe_receivers] - positions[safe_senders]
        r_sq = jnp.sum(rij * rij, axis=-1)
        valid, per_edge = lj_energy_terms(r_sq, graph.edge_mask, **params)
        per_edge = jnp.asarray(edge_energy_scale, dtype) * jnp.where(
            valid, per_edge, jnp.asarray(0.0, dtype)
        )
        return jnp.zeros((positions.shape[0],), dtype=dtype).at[safe_senders].add(per_edge)

    return lj_energy


def make_lj_edge_force(*, cutoff: float, epsilon: float, sigma: float):
    """Build a force_fn for the edge-force output contract.

    Returns per-edge dU/d(rij) with rij = x_receiver - x_sender, zero on masked edges.
    """
    params = dict(cutoff_sq=cutoff * cutoff, epsilon=epsilon, sigma_sq=sigma * sigma)

    def edge_energy_from_rij(rij):
        r_sq = jnp.sum(rij * rij, axis=-1)
        valid, energy = lj_energy_terms(r_sq, jnp.bool_(True), **params)
        return jnp.where(valid, energy, jnp.asarray(0.0, energy.dtype))

    edge_grad = jax.vmap(jax.grad(edge_energy_from_rij))

    def lj_edge_force(positions, species, graph):
        del species
        safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
        safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
        rij = positions[safe_receivers] - positions[safe_senders]
        edge_force = edge_grad(rij)
        return jnp.where(graph.edge_mask[:, None], edge_force, jnp.asarray(0.0, edge_force.dtype))

    return lj_edge_force


def add_shared_arguments(parser, *, max_atoms, edges_per_atom):
    parser.add_argument("output", help="Destination path for the JSON bundle.")
    parser.add_argument(
        "--max-atoms", type=int, default=max_atoms,
        help="Per-rank atom capacity, owned plus ghost rows; the run "
             "aborts when any rank exceeds it.",
    )
    parser.add_argument(
        "--max-edges", type=int, default=None,
        help="Per-rank edge capacity; default max-atoms * edges-per-atom.",
    )
    parser.add_argument(
        "--edges-per-atom", type=int, default=edges_per_atom,
        help="Edge capacity per atom row, used when --max-edges is unset.",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float32",
        help="Floating-point ABI of the exported programs.",
    )


def export_lj(args, max_edges):
    edge_energy_scale = 1.0 if args.force_output == "edge" and args.newton == "on" else 0.5
    energy_fn = make_lj_energy(
        cutoff=args.cutoff,
        epsilon=args.epsilon,
        sigma=args.sigma,
        edge_energy_scale=edge_energy_scale,
    )
    force_fn = (
        make_lj_edge_force(cutoff=args.cutoff, epsilon=args.epsilon, sigma=args.sigma)
        if args.force_output == "edge"
        else None
    )

    if args.uses_box:
        inner_energy, inner_force = energy_fn, force_fn

        def energy_with_box(positions, species, graph, box):
            del box  # LJ ignores the cell; this exercises the box ABI.
            return inner_energy(positions, species, graph)

        energy_fn = energy_with_box
        if inner_force is not None:

            def force_with_box(positions, species, graph, box):
                del box
                return inner_force(positions, species, graph)

            force_fn = force_with_box

    export_model(
        energy_fn=energy_fn,
        force_fn=force_fn,
        path=args.output,
        max_atoms=args.max_atoms,
        max_edges=max_edges,
        cutoff=args.cutoff,
        unit_style="lj",
        precision=args.precision,
        newton=args.newton,
        force_output="edge-force" if args.force_output == "edge" else "atom-force",
        uses_box=args.uses_box,
    )


def export_eam(args, max_edges, parser):
    communicating = args.mode == "comm"
    if args.setfl is not None and args.funcfl is not None:
        parser.error("--setfl and --funcfl are mutually exclusive")
    if args.setfl is not None or args.funcfl is not None:
        table_path = args.setfl if args.setfl is not None else args.funcfl
        tables = load_setfl(args.setfl) if args.setfl else load_funcfl(args.funcfl)
        print(f"{table_path}: elements {tables['elements']} "
              f"cutoff {tables['cutoff']:.6f}")
        energy_fn = make_setfl_energy(tables, communicating=communicating,
                                      half_edges=args.half_edges)
        cutoff = tables["cutoff"]
        unit_style = "metal"
        n_species = len(tables["elements"])
    else:
        energy_fn = make_eam_energy(
            cutoff=args.cutoff,
            pair_a=args.pair_a,
            dens_f0=args.dens_f0,
            embed_c=args.embed_c,
            pair_embedding=args.pair_embedding,
            communicating=communicating,
            half_edges=args.half_edges,
        )
        cutoff = args.cutoff
        unit_style = "lj"
        n_species = None
    export_model(
        energy_fn=energy_fn,
        path=args.output,
        max_atoms=args.max_atoms,
        max_edges=max_edges,
        cutoff=cutoff,
        unit_style=unit_style,
        precision=args.precision,
        comm=communicating,
        n_hops=1 if communicating else 2,
        half_edges=args.half_edges,
        n_species=n_species,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="model", required=True)

    lj = subparsers.add_parser("lj", help="Truncated 12-6, no shift or tail correction.")
    add_shared_arguments(lj, max_atoms=1024, edges_per_atom=96)
    lj.add_argument("--cutoff", type=float, default=2.5)
    lj.add_argument("--epsilon", type=float, default=1.0)
    lj.add_argument("--sigma", type=float, default=1.0)
    lj.add_argument(
        "--force-output", choices=("atom", "edge"), default="edge",
        help="atom: forces by autodiff of the summed energy; edge: the "
             "model returns per-edge dU/d(rij).",
    )
    lj.add_argument(
        "--newton", choices=("on", "off"), default="off",
        help="Neighbor-list convention the bundle is exported for; the "
             "pair style enforces the matching LAMMPS setting at load.",
    )
    lj.add_argument(
        "--uses-box",
        action="store_true",
        help="Export with a trailing [3,3] cell-matrix input in the export precision.",
    )

    eam = subparsers.add_parser(
        "eam", help="Analytic Finnis-Sinclair or a tabulated DYNAMO potential.")
    add_shared_arguments(eam, max_atoms=4096, edges_per_atom=64)
    eam.add_argument(
        "--cutoff", type=float, default=1.6,
        help="Cutoff radius of the analytic model; tabulated exports take "
             "the file's cutoff.",
    )
    eam.add_argument(
        "--setfl", default=None,
        help="Path to a DYNAMO .eam.alloy file, optionally gzipped; replaces "
             "the analytic model and ignores --cutoff and the analytic "
             "parameters.",
    )
    eam.add_argument(
        "--funcfl", default=None,
        help="Path to a single-element DYNAMO funcfl .eam file "
             "(pair_style eam); same effect as --setfl.",
    )
    eam.add_argument("--pair-a", type=float, default=1.0, help="Analytic pair amplitude A.")
    eam.add_argument("--dens-f0", type=float, default=1.0, help="Analytic density prefactor f0.")
    eam.add_argument("--embed-c", type=float, default=1.5, help="Analytic embedding strength C.")
    eam.add_argument(
        "--pair-embedding", type=float, default=0.0,
        help="Neighbor-embedding cross term; gives the energy itself a "
             "two-cutoff receptive field.",
    )
    eam.add_argument(
        "--mode", choices=("ghost", "comm"), default="ghost",
        help="ghost: extended ghost shell, no in-program exchange. "
             "comm: one-cutoff shell with a density exchange, mirroring "
             "native pair_eam.",
    )
    eam.add_argument(
        "--half-edges", action="store_true",
        help="Pack each pair once instead of both directions; size "
             "--edges-per-atom for the deduplicated count.",
    )

    args = parser.parse_args()
    if args.precision == "float64":
        # Nothing has been traced yet, so config.update still applies.
        jax.config.update("jax_enable_x64", True)
    max_edges = (
        args.max_edges
        if args.max_edges is not None
        else args.max_atoms * args.edges_per_atom
    )
    if args.model == "lj":
        export_lj(args, max_edges)
    else:
        export_eam(args, max_edges, parser)


if __name__ == "__main__":
    main()
