"""Export a tabulated EAM bundle running Pallas kernels under `pair_style jax/kk`, float32.

By default forces come from lammps_jax.eam_pallas edge kernels over the compacted half edges and
energies from lammps_jax.eam.make_setfl_energy. With --neighbor-matrix both run as row kernels
over the LAMMPS list itself, one lane per atom: the half list with newton on, or with --newton
off the full list, where rows are complete and no atomics reach the neighbors. The density is
exchanged in the model.
"""

import argparse

import jax._src.lib

from lammps_jax.eam import load_setfl, make_setfl_energy
from lammps_jax.eam_pallas import (make_setfl_pallas_force, make_setfl_pallas_row_energy,
                                   make_setfl_pallas_row_force, relaxed_atomics)
from lammps_jax.export import export_model

# Later jaxlibs compile Triton ahead of time for a guessed GPU and use another custom call.
TRITON_CUSTOM_CALL_JAXLIB = (0, 10, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("setfl", help="DYNAMO .eam.alloy file, optionally gzipped.")
    parser.add_argument("output", help="Destination path for the JSON bundle.")
    parser.add_argument("--max-atoms", type=int, default=4096,
                        help="Per-rank atom capacity, owned plus ghost rows.")
    parser.add_argument("--max-edges", type=int, default=None,
                        help="Per-rank half-edge capacity; default max-atoms * edges-per-atom.")
    parser.add_argument("--edges-per-atom", type=int, default=32,
                        help="Half-edge capacity per atom row when --max-edges is not given.")
    parser.add_argument("--neighbor-matrix", type=int, default=None, metavar="SLOTS",
                        help="Take the LAMMPS half list as a matrix with this many slots per "
                             "row and run one lane per atom instead of packing edges.")
    parser.add_argument("--max-owned", "--owned-rows", dest="max_owned", type=int, default=None,
                        help="Matrix rows, the per-rank owned-atom capacity; default max-atoms.")
    parser.add_argument("--block", type=int, default=None,
                        help="Edges or rows per kernel program, a power of two; default 1024 "
                             "edges, or rows // 128 rounded down to a power of two within 32 "
                             "to 512.")
    parser.add_argument("--num-warps", type=int, default=None,
                        help="Warps per program, a power of two; default 4, or one per 32 rows.")
    parser.add_argument("--newton", choices=("on", "off"), default="on",
                        help="LAMMPS newton pair setting the bundle runs under; off takes the "
                             "full list and needs --neighbor-matrix.")
    args = parser.parse_args()
    matrix = args.neighbor_matrix is not None
    half_edges = args.newton == "on"
    if not matrix and args.max_owned is not None:
        parser.error("--max-owned sizes the neighbor matrix; give --neighbor-matrix too")
    if not matrix and not half_edges:
        parser.error("--newton off runs row kernels over the full list; give --neighbor-matrix")
    if matrix and args.max_edges is not None:
        parser.error("--max-edges belongs to the edge layout; drop it or --neighbor-matrix")
    if matrix:
        rows = args.max_owned or args.max_atoms
        block = args.block or min(512, max(32, 1 << max(0, (rows // 128).bit_length() - 1)))
        num_warps = args.num_warps or max(1, block // 32)
    else:
        block = args.block or 1024
        num_warps = args.num_warps or 4
    if jax._src.lib.version > TRITON_CUSTOM_CALL_JAXLIB:
        parser.error("this export path was validated with jaxlib <= 0.10.1, which embeds the "
                     "kernel as the __gpu$xla.gpu.triton custom call compiled by the plugin")

    tables = load_setfl(args.setfl)
    print(f"{args.setfl}: elements {tables['elements']} cutoff {tables['cutoff']:.6f}")
    if matrix:
        energy_fn = make_setfl_pallas_row_energy(tables, block=block, num_warps=num_warps,
                                                 half_edges=half_edges)
        force_fn = make_setfl_pallas_row_force(tables, block=block, num_warps=num_warps,
                                               half_edges=half_edges)
        capacity = dict(max_neighbors=args.neighbor_matrix, max_owned=args.max_owned)
    else:
        energy_fn = make_setfl_energy(tables, communicating=True, half_edges=True)
        force_fn = make_setfl_pallas_force(tables, block=block, num_warps=num_warps)
        capacity = dict(max_edges=args.max_edges or args.max_atoms * args.edges_per_atom)
    with relaxed_atomics():
        export_model(
            energy_fn=energy_fn,
            force_fn=force_fn,
            path=args.output,
            max_atoms=args.max_atoms,
            **capacity,
            cutoff=tables["cutoff"],
            unit_style="metal",
            force_output="atom-force",
            newton=args.newton,
            comm=True,
            half_edges=half_edges,
            n_species=len(tables["elements"]),
            custom_call_targets=("__gpu$xla.gpu.triton",),
        )


if __name__ == "__main__":
    main()
