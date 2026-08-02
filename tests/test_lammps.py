"""Integration tests running `pair_style jax/kk` through real LAMMPS.

Runs each deck on 1 and 2 MPI ranks against dense float64 references.
Skipped unless LAMMPS_BIN, PJRT_PLUGIN, and LAMMPS_PLUGIN_PATH are set.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from helpers import native_interpolate
from lammps_jax.eam import load_setfl, spline_lookup

REPO_ROOT = Path(__file__).resolve().parent.parent
CUZR_SETFL = REPO_ROOT / "examples" / "potentials" / "CuZr.eam.alloy.gz"
MACE_MP_DIR = Path(os.environ.get(
    "MACE_MP_BUNDLE_DIR", REPO_ROOT.parent / "models" / "mace-mp-0-small-jax"))

requires_lammps = pytest.mark.skipif(
    not all(os.environ.get(name)
            for name in ("LAMMPS_BIN", "PJRT_PLUGIN", "LAMMPS_PLUGIN_PATH")),
    reason="set LAMMPS_BIN, PJRT_PLUGIN, and LAMMPS_PLUGIN_PATH to run "
           "the LAMMPS integration tests",
)

MAX_RANK_FORCE_ERROR = 1.0e-5
MAX_RANK_POSITION_ERROR = 1.0e-6
MAX_RANK_ENERGY_ERROR_PER_ATOM = 5.0e-5
MAX_RANK_PRESSURE_ERROR = 1.0e-4
MAX_DENSE_FORCE_ERROR = 5.0e-5
MAX_DENSE_ENERGY_ERROR_PER_ATOM = 5.0e-6
MAX_TRACE_ERROR_PER_ATOM = 5.0e-6
MAX_NVE_DRIFT_PER_ATOM = 1.0e-4
MAX_TRAJECTORY_FORCE_ERROR = 5.0e-2
MAX_TRAJECTORY_POSITION_ERROR = 5.0e-3
MIN_FORCE_SIGNAL = 1.0e-6
MIN_PRESSURE_SIGNAL = 1.0e-3
MAX_F64_FORCE_ERROR = 1.0e-11
MAX_F64_ENERGY_ERROR_PER_ATOM = 1.0e-13
MAX_F64_PRESSURE_ERROR = 1.0e-13

EAM_CUTOFF = 1.6
EAM_PAIR_A = 1.0
EAM_DENS_F0 = 1.0
EAM_EMBED_C = 1.5
EAM_EMBED_EPS = 1.0e-3
EAM_PAIR_EMBED = 0.3

EXPORTS = {
    "lj": ("export_model.py",
           ("lj", "{output}", "--max-atoms", "2048", "--edges-per-atom", "96")),
    "lj_f64": ("export_model.py",
               ("lj", "{output}", "--max-atoms", "2048", "--edges-per-atom", "96",
                "--precision", "float64")),
    "eam": ("export_model.py",
            ("eam", "{output}", "--max-atoms", "4096", "--edges-per-atom", "64")),
    "eam_f64": ("export_model.py",
                ("eam", "{output}", "--max-atoms", "4096", "--edges-per-atom", "64",
                 "--precision", "float64")),
    "eam_comm": ("export_model.py",
                 ("eam", "{output}", "--max-atoms", "4096", "--edges-per-atom", "64",
                  "--mode", "comm", "--pair-embedding", str(EAM_PAIR_EMBED))),
    "eam_ghostx": ("export_model.py",
                ("eam", "{output}", "--max-atoms", "4096", "--edges-per-atom", "64",
                 "--pair-embedding", str(EAM_PAIR_EMBED))),
    "eam_comm_half": ("export_model.py",
                      ("eam", "{output}", "--max-atoms", "4096",
                       "--edges-per-atom", "64", "--mode", "comm",
                       "--pair-embedding", str(EAM_PAIR_EMBED),
                       "--half-edges")),
    "eam_small": ("export_model.py",
                  ("eam", "{output}", "--max-atoms", "512", "--edges-per-atom", "64")),
    "mace_comm": ("export_mace.py",
                  ("export", "plain", "{output}", "--mode", "comm", "--type-z", "13",
                   "--max-atoms", "2048", "--edges-per-atom", "16",
                   "--bundle-dir", str(MACE_MP_DIR), "--skip-check")),
    "mace_oeq": ("export_mace.py",
                 ("export", "oeq", "{output}", "--mode", "comm", "--type-z", "13",
                  "--max-atoms", "2048", "--edges-per-atom", "16",
                  "--bundle-dir", str(MACE_MP_DIR) + "-fused", "--skip-check")),
    "cuzr": ("export_model.py",
             ("eam", "{output}", "--setfl", "examples/potentials/CuZr.eam.alloy.gz",
              "--max-atoms", "8192", "--edges-per-atom", "128",
              "--precision", "float64")),
    "cuzr_edge": ("export_model.py",
                  ("eam", "{output}", "--setfl",
                   "examples/potentials/CuZr.eam.alloy.gz",
                   "--max-atoms", "3072", "--edges-per-atom", "32",
                   "--half-edges", "--mode", "comm", "--force-output", "edge")),
    "cuzr_half": ("export_model.py",
                  ("eam", "{output}", "--setfl", "examples/potentials/CuZr.eam.alloy.gz",
                   "--max-atoms", "8192", "--edges-per-atom", "64",
                   "--half-edges", "--precision", "float64")),
}

DECKS = {
    "lj_static": """\
variable bundle index lj.lammps-jax.json
variable dump_path index lj_static.dump

units lj
atom_style atomic
boundary p p p
newton off

lattice sc 0.8
region box block 0 6 0 6 0 6
create_box 1 box
create_atoms 1 box
mass 1 1.0

displace_atoms all random 0.05 0.05 0.05 12345 units box

neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes one 256 page 500000

pair_style jax/kk ${pjrt}
pair_coeff * * ${bundle}

thermo 1
thermo_style custom step atoms pe
thermo_modify norm no format float %.16g

dump forces all custom 1 ${dump_path} id type x y z fx fy fz
dump_modify forces sort id format float %.16g first yes

run_style verlet/kk
run 0
""",
    "eam_static": """\
variable bundle index eam.lammps-jax.json
variable dump_path index eam_static.dump
variable newton_setting index on

units lj
atom_style atomic
boundary p p p
newton ${newton_setting}

lattice fcc 0.8
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0

displace_atoms all random 0.03 0.03 0.03 12345 units box

neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes one 256 page 500000

pair_style jax/kk ${pjrt}
pair_coeff * * ${bundle}

thermo 1
thermo_style custom step atoms pe press
thermo_modify norm no format float %.16g

dump forces all custom 1 ${dump_path} id type x y z fx fy fz
dump_modify forces sort id format float %.16g first yes

run_style verlet/kk
run 0
""",
    "eam_nve": """\
variable bundle index eam.lammps-jax.json
variable dump_path index eam_nve.dump

units lj
atom_style atomic
boundary p p p
newton on

lattice fcc 0.8
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0

displace_atoms all random 0.03 0.03 0.03 12345 units box

neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes one 256 page 500000

pair_style jax/kk ${pjrt}
pair_coeff * * ${bundle}

velocity all create 0.6 4928459 loop geom
fix integrate all nve
timestep 0.002

thermo 5
thermo_style custom step atoms pe ke etotal press
thermo_modify norm no format float %.16g

dump trajectory all custom 5 ${dump_path} id type xu yu zu fx fy fz
dump_modify trajectory sort id format float %.16g first yes

run_style verlet/kk
run 0
run 50
""",
    "mace_nve": """\
variable bundle index mace.lammps-jax.json
variable dump_path index mace_nve.dump

units metal
atom_style atomic
boundary p p p
newton on

lattice fcc 4.05
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 26.9815

displace_atoms all random 0.05 0.05 0.05 12345 units box

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes one 512 page 500000

pair_style jax/kk ${pjrt}
pair_coeff * * ${bundle}

velocity all create 300.0 4928459 loop geom
fix integrate all nve
timestep 0.002

thermo 5
thermo_style custom step atoms pe ke etotal press
thermo_modify norm no format float %.16g

dump trajectory all custom 5 ${dump_path} id type xu yu zu fx fy fz
dump_modify trajectory sort id format float %.16g first yes

run_style verlet/kk
run 0
run 50
""",
    "cuzr_static": """\
variable bundle index cuzr.lammps-jax.json
variable dump_path index cuzr_static.dump

units metal
atom_style atomic
boundary p p p
newton on

lattice bcc 3.26
region box block 0 6 0 6 0 6
create_box 2 box
create_atoms 1 box basis 1 1 basis 2 2
mass 1 63.546
mass 2 91.224

displace_atoms all random 0.1 0.1 0.1 12345 units box

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes one 512 page 500000

pair_style jax/kk ${pjrt}
pair_coeff * * ${bundle}

thermo 1
thermo_style custom step atoms pe press
thermo_modify norm no format float %.16g

dump forces all custom 1 ${dump_path} id type x y z fx fy fz
dump_modify forces sort id format float %.16g first yes

run_style verlet/kk
run 0
""",
    "cuzr_types3_static": """\
variable bundle index cuzr.lammps-jax.json
variable dump_path index cuzr_types3.dump

units metal
atom_style atomic
boundary p p p
newton on

lattice bcc 3.26
region box block 0 6 0 6 0 6
create_box 3 box
create_atoms 1 box basis 1 1 basis 2 2
mass 1 63.546
mass 2 91.224
mass 3 91.224

pair_style jax/kk ${pjrt}
pair_coeff * * ${bundle}

run_style verlet/kk
run 0
""",
}

CASES = {
    "lj_static": dict(
        kind="static", deck="lj_static", bundle="lj", newton="off"),
    "lj_static_f64": dict(
        kind="static", deck="lj_static", bundle="lj_f64", newton="off",
        float64=True, dense="lj"),
    "eam_static_f64": dict(
        kind="static", deck="eam_static", bundle="eam_f64", newton="on",
        pressure=True, float64=True, dense="eam"),
    "eam_comm_static": dict(
        kind="static", deck="eam_static", bundle="eam_comm", newton="on",
        pressure=True, dense="eam_embedded"),
    "eam_ghostx_static": dict(
        kind="static", deck="eam_static", bundle="eam_ghostx", newton="on",
        pressure=True, dense="eam_embedded"),
    "eam_comm_half_static": dict(
        kind="static", deck="eam_static", bundle="eam_comm_half", newton="on",
        pressure=True, dense="eam_embedded"),
    "cuzr_static": dict(
        kind="static", deck="cuzr_static", bundle="cuzr", newton="on",
        pressure=True, float64=True, dense="cuzr",
        energy_tol=1.0e-12, pressure_tol=1.0e-8),
    "cuzr_half_static": dict(
        kind="static", deck="cuzr_static", bundle="cuzr_half", newton="on",
        pressure=True, float64=True, dense="cuzr",
        energy_tol=1.0e-12, pressure_tol=1.0e-8),
    "cuzr_edge_static": dict(
        kind="static", deck="cuzr_static", bundle="cuzr_edge", newton="on",
        pressure=True, dense="cuzr", pressure_tol=5.0e-3),
    "eam_nve": dict(kind="nve", deck="eam_nve", bundle="eam", dense="eam"),
    "mace_comm_nve": dict(kind="nve", deck="mace_nve", bundle="mace_comm",
                          pressure_tol=5.0e-2),
    "mace_oeq_nve": dict(kind="nve", deck="mace_nve", bundle="mace_oeq",
                         pressure_tol=5.0e-2),
}
STATIC_CASES = [name for name, spec in CASES.items() if spec["kind"] == "static"]
NVE_CASES = [name for name, spec in CASES.items() if spec["kind"] == "nve"]

NEGATIVE_CONTROLS = {
    "eam_newton_off": dict(bundle="eam", newton="off",
                           variables={"newton_setting": "off"},
                           message="exported for newton pair on"),
    "eam_capacity": dict(bundle="eam_small", newton="on", variables={},
                         message="capacity exceeded"),
    "cuzr_species": dict(bundle="cuzr", newton="on", variables={},
                         deck="cuzr_types3_static",
                         message="distinguishes 2 species"),
}

FORCE_COLUMNS = ("fx", "fy", "fz")
POSITION_TRIPLES = (("x", "y", "z"), ("xu", "yu", "zu"))


def position_triple(frame):
    """The dump frame's position columns: wrapped x/y/z or unwrapped xu/yu/zu."""
    for names in POSITION_TRIPLES:
        if all(name in frame for name in names):
            return names
    raise ValueError("dump frame has neither x/y/z nor xu/yu/zu columns")


# LAMMPS output parsing


def read_lammps_dump(path):
    """Parse a LAMMPS custom dump into a list of per-frame dicts.

    Each frame maps ATOMS columns to float64 arrays sorted by atom id,
    plus "box" as a (3, 2) lo/hi array and "timestep".
    """
    lines = Path(path).read_text().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        timestep = int(float(lines[i + 1].split()[0]))
        if not lines[i + 2].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"{path}: expected NUMBER OF ATOMS at line {i + 3}")
        natoms = int(lines[i + 3].split()[0])
        if not lines[i + 4].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"{path}: expected BOX BOUNDS at line {i + 5}")
        box = np.array(
            [[float(tok) for tok in lines[i + 5 + d].split()[:2]] for d in range(3)]
        )
        atoms_header = lines[i + 8]
        if not atoms_header.startswith("ITEM: ATOMS"):
            raise ValueError(f"{path}: expected ATOMS at line {i + 9}")
        columns = atoms_header.split()[2:]
        rows = np.array(
            [[float(tok) for tok in lines[i + 9 + a].split()] for a in range(natoms)]
        ).reshape(natoms, len(columns))
        order = np.argsort(rows[:, columns.index("id")])
        rows = rows[order]
        frame = {name: rows[:, c] for c, name in enumerate(columns)}
        frame["timestep"] = timestep
        frame["box"] = box
        frames.append(frame)
        i += 9 + natoms
    return frames


def parse_thermo_runs(text):
    """Extract thermo tables from LAMMPS screen output, one per `run`.

    A table starts at a "Step ..." header and ends at "Loop time";
    non-numeric lines inside a table are skipped.
    """
    runs = []
    current = None
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "Step":
            current = {"columns": tokens, "rows": []}
            runs.append(current)
            continue
        if line.startswith("Loop time"):
            current = None
            continue
        if current is None:
            continue
        try:
            values = [float(tok) for tok in tokens]
        except ValueError:
            continue
        if len(values) == len(current["columns"]):
            current["rows"].append(values)
    tables = []
    for run in runs:
        rows = np.array(run["rows"], dtype=np.float64).reshape(
            len(run["rows"]), len(run["columns"])
        )
        tables.append(
            {name: rows[:, idx] for idx, name in enumerate(run["columns"])}
        )
    return tables


def first_thermo_value(text, column):
    """Value of `column` in the first thermo row, the frame-0 state."""
    for run in parse_thermo_runs(text):
        if column in run and run[column].size:
            return float(run[column][0])
    raise ValueError(f"no thermo column {column!r} found in screen output")


def total_energy_trace_error(text_a, text_b, atoms):
    """Max per-atom deviation between the two variants' TotEng traces.

    Each trace is re-based to its first entry, so a constant offset
    between decompositions does not count.
    """
    traces = []
    for text in (text_a, text_b):
        parts = [run["TotEng"] for run in parse_thermo_runs(text) if "TotEng" in run]
        if not parts:
            raise ValueError("no thermo column 'TotEng' found in screen output")
        traces.append(np.concatenate(parts))
    trace_a, trace_b = traces
    if trace_a.size != trace_b.size:
        raise ValueError(
            f"thermo traces are not aligned: {trace_a.size} vs {trace_b.size} rows"
        )
    return float(np.max(np.abs((trace_b - trace_b[0]) - (trace_a - trace_a[0])))) / atoms


def maximum_segmented_nve_drift(text, atoms):
    """Max per-atom |TotEng(t) - TotEng(0)| across a screen's thermo rows.

    The baseline restarts wherever Step stops increasing: successive
    `run` commands and step resets each get their own baseline.
    """
    runs = [run for run in parse_thermo_runs(text) if "Step" in run and "TotEng" in run]
    if not runs:
        raise ValueError("no Step/TotEng thermo rows found in screen output")
    steps = np.concatenate([run["Step"] for run in runs])
    toteng = np.concatenate([run["TotEng"] for run in runs])
    worst = 0.0
    start = 0
    for k in range(1, steps.size + 1):
        if k == steps.size or steps[k] <= steps[k - 1]:
            segment = toteng[start:k]
            if segment.size:
                worst = max(worst, float(np.max(np.abs(segment - segment[0]))))
            start = k
    return worst / atoms


# Frame comparison


def max_component_difference(frame_a, frame_b, columns):
    return max(
        float(np.max(np.abs(frame_a[name] - frame_b[name]))) for name in columns
    )


def compare_rank_runs(frames_a, frames_b, trajectory=False):
    """Compare two dump-frame lists of the same deck, np1 vs np2.

    Frame 0 is a strict same-coordinates comparison; trajectory=True
    adds loose over-all-frames maxima.
    """
    if len(frames_a) != len(frames_b):
        raise ValueError(
            f"frame count mismatch: {len(frames_a)} vs {len(frames_b)}"
        )
    if not frames_a:
        raise ValueError("no frames to compare")
    for frame_a, frame_b in zip(frames_a, frames_b):
        if not np.array_equal(frame_a["id"], frame_b["id"]):
            raise ValueError("atom ids differ between runs")
        if not np.array_equal(frame_a["type"], frame_b["type"]):
            raise ValueError("atom types differ between runs")
    pos_columns = position_triple(frames_a[0])
    metrics = {
        "frame0_force_signal": max(
            float(np.max(np.abs(frames_a[0][name]))) for name in FORCE_COLUMNS
        ),
        "frame0_force_error": max_component_difference(
            frames_a[0], frames_b[0], FORCE_COLUMNS
        ),
        "frame0_position_error": max_component_difference(
            frames_a[0], frames_b[0], pos_columns
        ),
    }
    if trajectory:
        metrics["trajectory_force_error"] = max(
            max_component_difference(fa, fb, FORCE_COLUMNS)
            for fa, fb in zip(frames_a, frames_b)
        )
        metrics["trajectory_position_error"] = max(
            max_component_difference(fa, fb, pos_columns)
            for fa, fb in zip(frames_a, frames_b)
        )
    return metrics


# Dense float64 references


def dense_value_and_grad(total_energy, frame):
    """Energy and forces of a dense total-energy function in float64.

    Pinned to CPU under a scoped x64 flag, so this process claims no
    GPU memory alongside the LAMMPS ranks.
    """
    box = np.asarray(frame["box"], dtype=np.float64)
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        lengths = jnp.asarray(box[:, 1] - box[:, 0])
        positions = jnp.stack(
            [
                jnp.asarray(np.asarray(frame[name], dtype=np.float64))
                for name in position_triple(frame)
            ],
            axis=1,
        )
        energy, gradient = jax.value_and_grad(
            lambda pos: total_energy(pos, lengths)
        )(positions)
        return float(energy), np.asarray(-gradient)


def dense_eam_reference(frame, pair_embedding=0.0):
    """Evaluate the analytic EAM model densely in float64.

    Mirrors python/lammps_jax/eam.py with all-pairs minimum-image
    distances, so the reference is decomposition-independent.
    """
    cutoff_sq = EAM_CUTOFF * EAM_CUTOFF

    def total_energy(pos, lengths):
        displacement = pos[:, None, :] - pos[None, :, :]
        displacement = displacement - lengths * jnp.round(displacement / lengths)
        r_sq = jnp.sum(displacement * displacement, axis=-1)
        off_diagonal = ~jnp.eye(pos.shape[0], dtype=bool)
        valid = off_diagonal & (r_sq < cutoff_sq)
        envelope = jnp.where(valid, (1.0 - r_sq / cutoff_sq) ** 2, 0.0)
        envelope_sum = jnp.sum(envelope, axis=1)
        pair_energy = 0.5 * EAM_PAIR_A * envelope_sum
        density = EAM_DENS_F0 * envelope_sum
        embedding = -EAM_EMBED_C * (
            jnp.sqrt(density + EAM_EMBED_EPS) - jnp.sqrt(EAM_EMBED_EPS)
        )
        energy = pair_energy + embedding
        if pair_embedding != 0.0:
            energy = energy + pair_embedding * jnp.sum(
                envelope * embedding[None, :], axis=1
            )
        return jnp.sum(energy)

    return dense_value_and_grad(total_energy, frame)


def dense_lj_reference(frame):
    """Dense float64 LJ mirroring export_model.py's lj defaults.

    Plain 12-6 with a hard cutoff at 2.5, epsilon = sigma = 1.
    """

    def total_energy(pos, lengths):
        displacement = pos[:, None, :] - pos[None, :, :]
        displacement = displacement - lengths * jnp.round(displacement / lengths)
        r_sq = jnp.sum(displacement * displacement, axis=-1)
        off_diagonal = ~jnp.eye(pos.shape[0], dtype=bool)
        valid = off_diagonal & (r_sq < 2.5 * 2.5)
        inv_r6 = jnp.where(valid, (1.0 / jnp.where(valid, r_sq, 1.0)) ** 3, 0.0)
        return jnp.sum(0.5 * 4.0 * (inv_r6 * inv_r6 - inv_r6))

    return dense_value_and_grad(total_energy, frame)


def dense_cuzr_reference(frame):
    """Dense float64 eam/alloy from the committed Zhou CuZr tables.

    Coefficients rebuilt by helpers.native_interpolate, so only raw table
    values are shared with the exported model.
    """
    tables = load_setfl(str(CUZR_SETFL))
    for family in ("density", "pair", "embedding"):
        tables[family] = np.stack(
            [native_interpolate(c[:, 0]) for c in tables[family]])
    species_np = np.asarray(frame["type"], dtype=np.int64) - 1
    n_atoms = species_np.size
    cutoff_sq = tables["cutoff"] ** 2

    def total_energy(pos, lengths):
        species = jnp.asarray(species_np, dtype=jnp.int32)
        density_t = jnp.asarray(tables["density"])
        pair_t = jnp.asarray(tables["pair"])
        embed_t = jnp.asarray(tables["embedding"])
        pair_index = jnp.asarray(tables["pair_index"])
        displacement = pos[:, None, :] - pos[None, :, :]
        displacement = displacement - lengths * jnp.round(displacement / lengths)
        r_sq = jnp.sum(displacement * displacement, axis=-1)
        valid = (~jnp.eye(n_atoms, dtype=bool)) & (r_sq < cutoff_sq)
        r = jnp.sqrt(jnp.where(valid, r_sq, 1.0))
        spec_i = jnp.broadcast_to(species[:, None], r.shape)
        spec_j = jnp.broadcast_to(species[None, :], r.shape)
        rho_edge = spline_lookup(density_t, spec_j, r, tables["dr"])
        density = jnp.sum(jnp.where(valid, rho_edge, 0.0), axis=1)
        z2 = spline_lookup(pair_t, pair_index[spec_i, spec_j], r, tables["dr"])
        pair_energy = 0.5 * jnp.sum(jnp.where(valid, z2 / r, 0.0), axis=1)
        return jnp.sum(
            pair_energy + spline_lookup(embed_t, species, density, tables["drho"],
                                        extrapolate=True)
        )

    return dense_value_and_grad(total_energy, frame)


DENSE_REFERENCES = {
    "lj": dense_lj_reference,
    "eam": dense_eam_reference,
    "eam_embedded": lambda frame: dense_eam_reference(frame, EAM_PAIR_EMBED),
    "cuzr": dense_cuzr_reference,
}


# Process plumbing


class LammpsRunner:
    """Exports bundles and runs LAMMPS rank pairs on demand, memoized."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.mpirun = os.environ.get("MPIRUN", "mpirun")
        self.lammps_bin = os.environ["LAMMPS_BIN"]
        self.pjrt = Path(os.environ["PJRT_PLUGIN"]).resolve()
        env = dict(os.environ)
        env["LAMMPS_PLUGIN_PATH"] = str(Path(env["LAMMPS_PLUGIN_PATH"]).resolve())
        env.setdefault("JAX_PLATFORMS", "cpu")
        env.setdefault("LAMMPS_JAX_MEM_FRACTION", "0.3")
        nvidia_root = self.pjrt.parents[2] / "nvidia"
        lib_dirs = sorted(str(path) for path in nvidia_root.glob("*/lib"))
        if lib_dirs:
            existing = env.get("LD_LIBRARY_PATH", "")
            if existing:
                lib_dirs.append(existing)
            env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)
        self.env = env
        self.oversubscribe = False
        try:
            probe = subprocess.run(
                [self.mpirun, "--version"], capture_output=True, text=True,
                check=False,
            )
        except OSError:
            pass
        else:
            banner = (probe.stdout + probe.stderr).lower()
            self.oversubscribe = "open mpi" in banner or "open-mpi" in banner
        self.decks = {}
        for name, text in DECKS.items():
            self.decks[name] = out_dir / f"{name}.lmp"
            self.decks[name].write_text(text)
        self.bundles = {}
        self.rank_runs = {}
        self.shim_path = None

    def run(self, name, command, env=None, expect_success=True):
        command = [str(part) for part in command]
        screen = self.out_dir / f"{name}.screen"
        print(f"running {name}")
        with open(screen, "w") as sink:
            result = subprocess.run(
                command,
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=self.env if env is None else env,
                cwd=REPO_ROOT,
                check=False,
            )
        if expect_success and result.returncode != 0:
            raise RuntimeError(
                f"{name} failed with exit code {result.returncode} "
                f"({' '.join(command)}); see {screen}"
            )
        if not expect_success and result.returncode == 0:
            raise AssertionError(
                f"{name} was expected to fail but exited 0 "
                f"({' '.join(command)}); see {screen}"
            )
        return screen.read_text()

    def shim(self):
        """The replay shim library, built on first use."""
        if self.shim_path is None:
            source = REPO_ROOT / "contrib" / "ffi-replay" / "ffi_replay_shim.c"
            path = self.out_dir / "ffi_replay_shim.so"
            try:
                build = subprocess.run(
                    ["gcc", "-shared", "-fPIC", "-O2", "-o", path, source,
                     "-ldl", "-lpthread", "-l:libffi.so.8"],
                    capture_output=True, text=True, check=False,
                )
            except OSError:
                pytest.skip("gcc not available")
            if build.returncode != 0:
                pytest.skip(
                    f"cannot build ffi_replay_shim.so: {build.stderr.strip()}")
            self.shim_path = path
        return self.shim_path

    def sidecar_env(self, bundle_path):
        """Environment from the bundle's .env sidecar, empty without one."""
        env_path = Path(f"{bundle_path}.env")
        if not env_path.exists():
            return {}
        values = {}
        for line in env_path.read_text().splitlines():
            key, sep, value = line.removeprefix("export ").partition("=")
            if not sep or not key:
                continue
            values[key] = value.strip("'").replace(
                "/path/ffi_replay_shim.so", str(self.shim()))
        return values

    def bundle(self, name):
        """The bundle path for `name`, exporting it on first use."""
        if name not in self.bundles:
            script, arguments = EXPORTS[name]
            if script == "export_mace.py":
                bundle_dir = MACE_MP_DIR
                if "--bundle-dir" in arguments:
                    bundle_dir = Path(arguments[arguments.index("--bundle-dir") + 1])
                if not (bundle_dir / "config.json").exists():
                    pytest.skip("converted MACE-MP bundle not available")
            if "oeq" in arguments and importlib.util.find_spec("openequivariance") is None:
                pytest.skip("openequivariance not available")
            path = self.out_dir / f"{name}.lammps-jax.json"
            env = dict(self.env)
            env["JAX_ENABLE_X64"] = "0"
            command = [sys.executable, REPO_ROOT / "examples" / script]
            command += [path if arg == "{output}" else arg for arg in arguments]
            self.run(f"export_{name}", command, env=env)
            self.bundles[name] = path
        return self.bundles[name]

    def lammps(self, name, deck, newton, nprocs, variables, extra_env=None,
               expect_success=True):
        command = [self.mpirun, "-np", str(nprocs)]
        if self.oversubscribe:
            command.append("--oversubscribe")
        neigh = "half" if newton == "on" else "full"
        command += [
            self.lammps_bin,
            "-k", "on", "g", "1",
            "-sf", "kk",
            "-pk", "kokkos", "newton", newton, "neigh", neigh,
            "-log", "none",
            "-in", deck,
            "-var", "pjrt", self.pjrt,
        ]
        for key, value in variables.items():
            command += ["-var", key, value]
        env = {**self.env, **extra_env} if extra_env else None
        return self.run(name, command, env=env, expect_success=expect_success)

    def rank_pair(self, case):
        """Frames and screens of `case` on 1 and 2 ranks, run on first use."""
        if case not in self.rank_runs:
            spec = CASES[case]
            frames = {}
            screens = {}
            for nprocs in (1, 2):
                tag = f"{case}_np{nprocs}"
                dump_path = self.out_dir / f"{tag}.dump"
                bundle_path = self.bundle(spec["bundle"])
                screens[nprocs] = self.lammps(
                    tag,
                    self.decks[spec["deck"]],
                    newton=spec.get("newton", "on"),
                    nprocs=nprocs,
                    variables={"bundle": bundle_path,
                               "dump_path": dump_path},
                    extra_env=self.sidecar_env(bundle_path),
                )
                grid = re.search(r"(\d+) by (\d+) by (\d+) MPI processor grid",
                                 screens[nprocs])
                assert grid is not None
                assert int(grid[1]) * int(grid[2]) * int(grid[3]) == nprocs
                frames[nprocs] = read_lammps_dump(dump_path)
            self.rank_runs[case] = (frames, screens)
        return self.rank_runs[case]


@pytest.fixture(scope="session")
def runner(tmp_path_factory):
    return LammpsRunner(tmp_path_factory.mktemp("lammps"))


# Integration tests


@requires_lammps
@pytest.mark.parametrize("case", STATIC_CASES)
def test_static_rank_invariance(runner, case):
    spec = CASES[case]
    float64 = spec.get("float64", False)
    frames, screens = runner.rank_pair(case)
    natoms = frames[1][0]["id"].size
    metrics = compare_rank_runs(frames[1], frames[2])
    assert metrics["frame0_force_signal"] >= MIN_FORCE_SIGNAL
    assert metrics["frame0_force_error"] <= (
        MAX_F64_FORCE_ERROR if float64
        else spec.get("force_tol", MAX_RANK_FORCE_ERROR))
    assert metrics["frame0_position_error"] <= MAX_RANK_POSITION_ERROR
    energy_tol = spec.get("energy_tol",
                          MAX_F64_ENERGY_ERROR_PER_ATOM if float64
                          else MAX_RANK_ENERGY_ERROR_PER_ATOM)
    potential = {n: first_thermo_value(screens[n], "PotEng") for n in (1, 2)}
    assert abs(potential[1] - potential[2]) / natoms <= energy_tol
    if spec.get("pressure"):
        press = {n: first_thermo_value(screens[n], "Press") for n in (1, 2)}
        assert abs(press[1]) >= MIN_PRESSURE_SIGNAL
        assert abs(press[1] - press[2]) <= spec.get(
            "pressure_tol",
            MAX_F64_PRESSURE_ERROR if float64 else MAX_RANK_PRESSURE_ERROR)
    if spec.get("dense"):
        energy, forces = DENSE_REFERENCES[spec["dense"]](frames[1][0])
        dumped = np.stack([frames[1][0][name] for name in FORCE_COLUMNS], axis=1)
        assert float(np.max(np.abs(forces - dumped))) <= (
            MAX_F64_FORCE_ERROR if float64 else MAX_DENSE_FORCE_ERROR)
        dense_energy_tol = spec.get("energy_tol",
                                    MAX_F64_ENERGY_ERROR_PER_ATOM if float64
                                    else MAX_DENSE_ENERGY_ERROR_PER_ATOM)
        assert abs(energy - potential[1]) / natoms <= dense_energy_tol


@requires_lammps
@pytest.mark.parametrize("case", NVE_CASES)
def test_nve_trajectory(runner, case):
    """NVE on 1 vs 2 ranks; frame 0 also serves as the static check.

    Frame 0 matches a `run 0` static evaluation; loop-geom velocities
    add only a decomposition-independent kinetic pressure term.
    """
    spec = CASES[case]
    frames, screens = runner.rank_pair(case)
    natoms = frames[1][0]["id"].size
    metrics = compare_rank_runs(frames[1], frames[2], trajectory=True)
    assert metrics["frame0_force_signal"] >= MIN_FORCE_SIGNAL
    assert metrics["frame0_force_error"] <= spec.get("force_tol",
                                                     MAX_RANK_FORCE_ERROR)
    assert metrics["frame0_position_error"] <= MAX_RANK_POSITION_ERROR
    potential = {n: first_thermo_value(screens[n], "PotEng") for n in (1, 2)}
    assert (abs(potential[1] - potential[2]) / natoms
            <= MAX_RANK_ENERGY_ERROR_PER_ATOM)
    press = {n: first_thermo_value(screens[n], "Press") for n in (1, 2)}
    assert abs(press[1] - press[2]) <= spec.get("pressure_tol",
                                                MAX_RANK_PRESSURE_ERROR)
    if spec.get("dense"):
        energy, forces = DENSE_REFERENCES[spec["dense"]](frames[1][0])
        dumped = np.stack([frames[1][0][name] for name in FORCE_COLUMNS], axis=1)
        assert float(np.max(np.abs(forces - dumped))) <= MAX_DENSE_FORCE_ERROR
        assert (abs(energy - potential[1]) / natoms
                <= MAX_DENSE_ENERGY_ERROR_PER_ATOM)
    assert metrics["trajectory_force_error"] <= MAX_TRAJECTORY_FORCE_ERROR
    assert metrics["trajectory_position_error"] <= MAX_TRAJECTORY_POSITION_ERROR
    assert (total_energy_trace_error(screens[1], screens[2], natoms)
            <= MAX_TRACE_ERROR_PER_ATOM)
    for nprocs in (1, 2):
        assert (maximum_segmented_nve_drift(screens[nprocs], natoms)
                <= MAX_NVE_DRIFT_PER_ATOM)


@requires_lammps
@pytest.mark.parametrize(
    "case_a, case_b",
    [("eam_comm_static", "eam_ghostx_static"),
     ("cuzr_half_static", "cuzr_static"),
     ("eam_comm_half_static", "eam_comm_static")],
    ids=["eam", "cuzr-pairing", "eam-comm-pairing"],
)
def test_scheme_cross_agreement(runner, case_a, case_b):
    """Same weights and coordinates under two schemes or edge packings.

    Compares the two single-rank runs; frame 0 is a strict
    same-coordinates comparison.
    """
    frames_a, _ = runner.rank_pair(case_a)
    frames_b, _ = runner.rank_pair(case_b)
    cross = compare_rank_runs(frames_b[1], frames_a[1])
    spec_a, spec_b = CASES[case_a], CASES[case_b]
    both_f64 = spec_a.get("float64") and spec_b.get("float64")
    assert cross["frame0_force_error"] <= (
        MAX_F64_FORCE_ERROR if both_f64
        else max(spec_a.get("force_tol", MAX_RANK_FORCE_ERROR),
                 spec_b.get("force_tol", MAX_RANK_FORCE_ERROR)))


@requires_lammps
@pytest.mark.parametrize("case", NEGATIVE_CONTROLS)
def test_negative_control(runner, case):
    """A deck that must abort with its documented error message."""
    spec = NEGATIVE_CONTROLS[case]
    screen = runner.lammps(
        case,
        runner.decks[spec.get("deck", "eam_static")],
        newton=spec["newton"],
        nprocs=2,
        variables={"bundle": runner.bundle(spec["bundle"]),
                   "dump_path": runner.out_dir / f"{case}.dump",
                   **spec["variables"]},
        expect_success=False,
    )
    assert spec["message"] in screen


# CPU unit tests for the parsers and dense references above


DUMP_TEXT = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 4.0
0.0 4.0
0.0 4.0
ITEM: ATOMS id type x y z fx fy fz
2 1 1.0 0.0 0.0 0.5 0.0 0.0
1 1 0.0 0.0 0.0 -0.5 0.0 0.0
"""


def test_read_lammps_dump_sorts_by_id(tmp_path):
    path = tmp_path / "test.dump"
    path.write_text(DUMP_TEXT)
    frames = read_lammps_dump(path)
    assert len(frames) == 1
    frame = frames[0]
    np.testing.assert_array_equal(frame["id"], [1.0, 2.0])
    np.testing.assert_array_equal(frame["fx"], [-0.5, 0.5])
    np.testing.assert_array_equal(frame["box"][:, 1], [4.0, 4.0, 4.0])


def test_compare_rank_runs_reports_frame0_errors(tmp_path):
    path = tmp_path / "test.dump"
    path.write_text(DUMP_TEXT)
    frames = read_lammps_dump(path)
    shifted = [dict(frames[0])]
    shifted[0]["fx"] = frames[0]["fx"] + 1e-5
    metrics = compare_rank_runs(frames, shifted)
    assert metrics["frame0_force_error"] == pytest.approx(1e-5)
    assert metrics["frame0_position_error"] == 0.0


def test_segmented_nve_drift_restarts_baseline():
    text = """Step TotEng
0 -1.00
5 -1.01
10 -1.02
Loop time of 1.0
Step TotEng
0 -2.00
5 -2.00
10 -2.005
Loop time of 1.0
"""
    drift = maximum_segmented_nve_drift(text, atoms=2)
    assert drift == pytest.approx(0.02 / 2)


def test_dense_eam_reference_isolated_atoms_zero():
    frame = {
        "box": np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]),
        "x": np.array([1.0, 6.0]),
        "y": np.array([1.0, 6.0]),
        "z": np.array([1.0, 6.0]),
    }
    energy, forces = dense_eam_reference(frame)
    assert energy == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(forces, 0.0)
