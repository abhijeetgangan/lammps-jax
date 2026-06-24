"""Export a sparse Lennard-Jones energy model for `pair_style jax/kk`."""

import argparse

import jax
import jax.numpy as jnp

from lammps_jax.export import export_model


def lj_energy_terms(r_sq, mask, *, cutoff_sq, epsilon, sigma_sq):
    valid = mask & (r_sq > jnp.float32(1.0e-12)) & (r_sq < jnp.float32(cutoff_sq))
    safe_r_sq = jnp.where(valid, r_sq, jnp.float32(1.0))
    inv_r2 = jnp.float32(sigma_sq) / safe_r_sq
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    energy = jnp.float32(4.0) * jnp.float32(epsilon) * (inv_r6 * inv_r6 - inv_r6)
    return valid, energy


def make_lj_energy(*, cutoff: float, epsilon: float, sigma: float, edge_energy_scale: float = 0.5):
    params = dict(cutoff_sq=cutoff * cutoff, epsilon=epsilon, sigma_sq=sigma * sigma)
    edge_energy_scale = jnp.float32(edge_energy_scale)

    def lj_energy(positions, species, graph):
        del species
        safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
        safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
        rij = positions[safe_receivers] - positions[safe_senders]
        r_sq = jnp.sum(rij * rij, axis=-1)
        valid, per_edge = lj_energy_terms(r_sq, graph.edge_mask, **params)
        per_edge = edge_energy_scale * jnp.where(valid, per_edge, jnp.float32(0.0))
        return jnp.zeros((positions.shape[0],), dtype=jnp.float32).at[safe_senders].add(per_edge)

    return lj_energy


def make_lj_edge_force(*, cutoff: float, epsilon: float, sigma: float):
    params = dict(cutoff_sq=cutoff * cutoff, epsilon=epsilon, sigma_sq=sigma * sigma)

    def edge_energy_from_rij(rij):
        r_sq = jnp.sum(rij * rij, axis=-1)
        valid, energy = lj_energy_terms(r_sq, jnp.bool_(True), **params)
        return jnp.where(valid, energy, jnp.float32(0.0))

    edge_grad = jax.vmap(jax.grad(edge_energy_from_rij))

    def lj_edge_force(positions, species, graph):
        del species
        safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
        safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
        rij = positions[safe_receivers] - positions[safe_senders]
        edge_force = edge_grad(rij)
        return jnp.where(graph.edge_mask[:, None], edge_force, jnp.float32(0.0))

    return lj_edge_force


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--max-atoms", type=int, default=1024)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--edges-per-atom", type=int, default=96)
    parser.add_argument("--cutoff", type=float, default=2.5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--force-output", choices=("atom", "edge"), default="edge")
    parser.add_argument("--newton", choices=("on", "off"), default="off")
    parser.add_argument(
        "--uses-box",
        action="store_true",
        help="Export with a trailing f32[3,3] cell-matrix input.",
    )
    args = parser.parse_args()
    max_edges = (
        args.max_edges
        if args.max_edges is not None
        else args.max_atoms * args.edges_per_atom
    )

    if args.force_output == "atom" and args.newton != "on":
        raise ValueError("autodiff atom-force exports require --newton on")

    edge_energy_scale = 1.0 if args.force_output == "edge" and args.newton == "on" else 0.5
    energy_fn = make_lj_energy(
        cutoff=args.cutoff,
        epsilon=args.epsilon,
        sigma=args.sigma,
        edge_energy_scale=edge_energy_scale,
    )
    newton_contract = args.newton if args.force_output == "edge" else "any"
    model_kwargs = {
        "path": args.output,
        "max_atoms": args.max_atoms,
        "max_edges": max_edges,
        "cutoff": args.cutoff,
        "unit_style": "lj",
        "newton": newton_contract,
        "energy_fn": energy_fn,
        "force_output": "edge-force" if args.force_output == "edge" else "atom-force",
    }
    if args.force_output == "edge":
        model_kwargs["force_fn"] = make_lj_edge_force(
            cutoff=args.cutoff,
            epsilon=args.epsilon,
            sigma=args.sigma,
        )

    if args.uses_box:
        model_kwargs["uses_box"] = True

        def with_box(fn):
            def wrapped(positions, species, graph, box):
                del box  # LJ ignores the cell; this exercises the box ABI.
                return fn(positions, species, graph)

            return wrapped

        model_kwargs["energy_fn"] = with_box(model_kwargs["energy_fn"])

    export_model(**model_kwargs)


if __name__ == "__main__":
    main()
