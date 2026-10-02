"""
User-editable project settings: CAD source file and boundary-condition face
assignments.

Workflow:
    1. Put your STEP export from SolidWorks somewhere under cad_models/ (or
       anywhere) and point STEP_FILE at it.
    2. Run `python main.py --show` and rotate the model to read off the
       integer Face ID printed on each surface.
    3. Fill in CLAMP / INPUT / OUTPUT below with those IDs. Every other face
       is automatically treated as DESIGN (free) surface unless you list
       DESIGN_FACES explicitly.
    4. Run `python main.py --verify` to see the assignment colour-coded and
       confirm you picked the right surfaces before doing anything else.
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

# Where the original, verified nonlinear solver lives (never edited by this
# project -- Stage 5 imports it and subclasses its Model instead).
TOP3D_CF_DIR = PROJECT_DIR.parent / "files"

# Path to the CAD file (STEP/STP). The bundled sample is a simple mounting
# block with a through-hole, just to exercise the pipeline -- replace this
# with your own SolidWorks export.
STEP_FILE = PROJECT_DIR / "cad_models" / "Part1.step"

# --- Boundary-condition face assignments -----------------------------------
# The values below are for Part1.step. Re-run `python main.py --list` and
# update these any time the CAD geometry or face numbering changes -- Gmsh
# can renumber faces after even a small edit upstream in SolidWorks.
#
# INPUT and OUTPUT are LISTS OF GROUPS. Each group is some faces acting
# along ONE signed direction, and every group can have its own direction:
#
#   direction      "x", "y", "z", optionally "+x" / "-x" etc.
#
#   INPUT group    "displacement": m moved along `direction` at full stroke.
#                  "-x" with 0.004 and "x" with -0.004 both mean "pushed
#                  4 mm towards -x". All input groups ramp together, in
#                  proportion, over the nstep load steps.
#
#   OUTPUT group   "direction": the way this group's motion counts as
#                  POSITIVE output force. "spring_stiffness": N/m, a grounded
#                  spring (the load, e.g. tissue) on this group's faces.
#                  The optimiser flattens the TOTAL over all groups:
#                      Fout = sum over groups of (spring force along its
#                             own direction)
#                  so two jaws closing on each other -- one group "-y", the
#                  other "+y" -- both count positive. For a symmetric
#                  two-jaw gripper each jaw carries Fout / 2.
#
#   "name" (optional) labels a group in the printed reports.
#
# A plain dict instead of a list still works (= one group).
#
# Check the directions BEFORE optimising:
#   python voxel_main.py --show        an arrow per group shows its direction
#   python linear_check_main.py        per-group displacement + the sign of
#                                      Fout; it warns if the outputs move
#                                      AGAINST their configured directions
# If Fout comes out negative there and Ftar is None, the optimiser takes
# its target from that starting response and flattens a force in the
# OPPOSITE direction. Fix the directions, or set OPTIMIZATION["Ftar"] > 0.
#
# Reminder from top3d_cf.py's README: input and output groups that are only
# weakly coupled through the body (e.g. far apart, or crossing directions on
# a uniform start) give a sign-indefinite initial force path and the
# optimiser will refuse to start.

CLAMP = {
    "faces": [19,23],              # face(s) bolted/mounted -- fully fixed
    "dofs": ["x", "y", "z"],
}

INPUT = [
    {"name": "actuator", "faces": [15], "direction": "x", "displacement": 0.001},
]

OUTPUT = [
    {"name": "upper_jaw", "faces": [8], "direction": "-y", "spring_stiffness": 1000.0},
     {"name": "lower_jaw", "faces": [10], "direction": "+y", "spring_stiffness": 1000.0}
]

# Example: the same part as a gripper whose jaws close on each other. From
# `main.py --list`, face 8 sits at y = +0.0025 and face 10 at y = -0.0025, so
# closing is face 8 moving -y and face 10 moving +y. Confirm with
# `voxel_main.py --show` that the two output arrows point at each other.
#
# OUTPUT = [
#     {"name": "upper jaw", "faces": [8],  "direction": "-y", "spring_stiffness": 500.0},
#     {"name": "lower jaw", "faces": [10], "direction": "+y", "spring_stiffness": 500.0},
# ]

# Faces available to the optimiser as free/design surface.
# None = "every face not already claimed by CLAMP/INPUT/OUTPUT".
DESIGN_FACES = None

# --- Voxelization (Stage 3) --------------------------------------------------
# Elements are cubes; this sets how many fit along the longest bounding-box
# dimension (the other two axes get whatever whole number of cubes fits).
# Higher = finer boundary resolution and exact face-matching, at the cost of
# more elements. Compare to top3d_cf.py's own default nelx=32.
VOXEL_NEL_LONG_AXIS = 24

# Extra empty background-grid layers padded around the CAD bounding box.
VOXEL_PAD_ELEMENTS = 1

# --- Material (Stage 4 linear check, and Stage 5's E0/nu) -------------------
# Defaults match top3d_cf.py's own Params (E0, nu).
MATERIAL = {
    "E": 2.0e9  ,   # Pa
    "nu": 0.40,
}

# --- Topology optimisation (Stage 5) -----------------------------------------
# Passed straight through to top3d_cf.Params -- see that file for what each
# one does. uin/kout are NOT here: they come from INPUT["displacement"] and
# OUTPUT["spring_stiffness"] above, so the load case only has to be written
# down once.
OPTIMIZATION = {
    "volfrac": 0.30,
    "penal": 3.0,
    "rmin": 2.0,        # filter radius, in elements
    "move": 0.05,
    "maxiter": 100,
    "nstep": 10,        # load steps over the input stroke
    "nrise": 3,          # steps 1..nrise are the preload/rise region
    "Ftar": None,         # target plateau force, N; None = auto (median of the first pass)
}

# Design update used by optimize_main.py --run / --time:
#   "mma"  Svanberg's GCMMA (optimization/mma.py): MMA that checks every
#          step against a fresh force-path solve and shortens the ones
#          that overshoot, so the design settles instead of swinging.
#   "pg"   top3d_cf.py's own projected-gradient step. Every iteration moves
#          the most sensitive elements by the full "move" above, so it
#          flip-flops instead of settling; kept only for comparison.
# Either way "move" is the largest change of any element in one iteration.
OPTIMIZER = "mma"
