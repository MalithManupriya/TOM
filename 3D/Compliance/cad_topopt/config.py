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
# "faces" (a list of integer Face IDs) is the only key Stage 1/2 actually
# uses right now. The remaining keys (direction, displacement, dofs,
# spring_stiffness) are not read yet -- they are for Stage 5, when these
# boundary conditions get wired into the nonlinear solver from top3d_cf.py.
# They are defined here now so the config format/shape does not change later
# and so you can see the full intended meaning of each face group up front.
#
# The values below are for Part1.step. Re-run `python main.py --show` and
# update these any time the CAD geometry or face numbering changes -- Gmsh
# can renumber faces after even a small edit upstream in SolidWorks.
# Reminder from top3d_cf.py's README: input and output should sit on the
# same free end, acting along the same axis, or the initial force path comes
# out sign-indefinite and the optimiser will refuse to start.

CLAMP = {
    "faces": [9],              # face(s) bolted/mounted -- fully fixed
    "dofs": ["x", "y", "z"],
}

INPUT = {
    "faces": [15],                # face(s) pushed by the actuator
    "direction": "x",
    "displacement": -0.004,     # m, total prescribed stroke (Stage 5)
}

OUTPUT = {
    "faces": [8,10],                 # face(s) that contact the load (e.g. tissue)
    "direction": "x",
    "spring_stiffness": 1000.0,  # N/m, grounded output spring (Stage 5)
}

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
    "E": 2.0e9,   # Pa
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
