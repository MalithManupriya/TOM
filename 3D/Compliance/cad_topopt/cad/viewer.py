"""
Visual verification of face numbering and face-role assignment.

Both functions here require an already-open Gmsh session (i.e. call them
inside a `with CadModel(...) as cad:` block) and both block until the viewer
window is closed -- that's Gmsh's FLTK GUI, not a custom viewer.

Two things had to be set for the numbering to actually be readable, found by
rendering it and looking, not by guessing:
  - Gmsh draws each face's label in that face's own colour. Left alone, a
    STEP file that carries per-face appearance colours from SolidWorks (or
    Gmsh's own pale default) makes the text nearly invisible against a white
    background. So every face is given an explicit, high-contrast colour
    before the labels are drawn.
  - The default label is not just the number -- it's the full entity
    description plus a second line literally printing that face's RGB (e.g.
    "Plane 6 (OCC)" / "Color (202, 209, 238)"), which is what actually made
    the view unreadable, more than the colour did. Geometry.LabelType=1
    reduces it to the bare Face ID.
"""
from __future__ import annotations

import gmsh

from .face_manager import FaceAssignment

# RGB, 0-255. Chosen dark enough to stay legible as label text on a white
# background -- Gmsh draws each face's number in the face's own colour, so a
# pale colour (the ROLE_COLORS "design" grey used to be (190,190,190)) makes
# the label nearly invisible, independent of how it looks as an outline.
NEUTRAL_COLOR = (20, 20, 20)          # Stage 1: every face, before assignment
ROLE_COLORS = {
    "clamp": (190, 40, 40),   # fixed / mounted
    "input": (30, 130, 30),   # prescribed displacement
    "output": (30, 80, 180),  # spring / measured force
    "design": (90, 90, 90),   # free surface, left to the optimiser
}


def _use_plain_labels() -> None:
    gmsh.option.setNumber("Geometry.SurfaceNumbers", 1)
    gmsh.option.setNumber("Geometry.LabelType", 1)  # tag only, no description


def show_face_ids(label_faces: bool = True) -> None:
    """Stage 1: raw numbered view. Rotate the part and read the ID off each face."""
    if label_faces:
        faces = gmsh.model.occ.getEntities(dim=2)
        gmsh.model.setColor(faces, *NEUTRAL_COLOR, recursive=False)
        _use_plain_labels()
    gmsh.fltk.run()


def show_assignment(assignment: FaceAssignment) -> None:
    """Stage 2: colour every face by its assigned role, then open the viewer."""
    for tag, role in assignment.role_by_face.items():
        gmsh.model.setColor([(2, tag)], *ROLE_COLORS[role], recursive=False)
    _use_plain_labels()

    print("Viewer legend:")
    for role, rgb in ROLE_COLORS.items():
        faces = assignment.faces_with_role(role)
        print(f"  {role:8s} rgb{rgb}  faces={faces}")

    gmsh.fltk.run()
