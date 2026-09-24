"""
Load a STEP/IGES/BREP file and enumerate its faces.

Design choice: CAD import goes through Gmsh's built-in OpenCASCADE kernel
(`gmsh.model.occ`) instead of a separate CAD library such as OCP or
pythonocc-core. Gmsh is also going to do the meshing in Stage 3, and a face's
"Face ID" only means one consistent thing -- a Gmsh surface tag -- if the same
library both numbers the faces and later meshes them. Importing through a
second CAD library that numbers faces its own way would risk the numbering
you read off the viewer silently disagreeing with the numbering used to build
node sets later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh


@dataclass
class FaceInfo:
    tag: int  # Gmsh surface tag -- this is the "Face ID" shown to the user
    area: float
    centroid: tuple[float, float, float]
    bbox: tuple[float, float, float, float, float, float]  # xmin,ymin,zmin,xmax,ymax,zmax


class CadModel:
    """Thin wrapper around one Gmsh session holding one imported CAD solid.

    Gmsh keeps its model in a single global session, so this class does not
    let you hold two independent CAD models at once -- it is a convenience
    wrapper around that session, used as a context manager so
    gmsh.initialize()/finalize() are always paired:

        with CadModel(step_path) as cad:
            cad.print_faces()
    """

    def __init__(self, step_path: str | Path):
        self.step_path = Path(step_path)
        if not self.step_path.exists():
            raise FileNotFoundError(
                f"CAD file not found: {self.step_path}\n"
                "Export your SolidWorks part as STEP (.step/.stp) and point "
                "config.STEP_FILE at it."
            )

    def __enter__(self) -> "CadModel":
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add(self.step_path.stem)
        # Gmsh's STEP reader does NOT convert to metres by default -- it
        # reads the file's raw numbers as-is even though SolidWorks (and
        # most STEP files) declare millimetres in the header. Left alone,
        # a 10 mm part silently becomes a 10 m part everywhere downstream
        # (element size, volume, therefore stiffness -- off by 1e9 in
        # force). Setting the target unit makes Gmsh read the file's own
        # declared unit and convert to metres, whatever that unit is.
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.occ.importShapes(str(self.step_path))
        gmsh.model.occ.synchronize()
        if not gmsh.model.occ.getEntities(dim=2):
            raise RuntimeError(
                f"{self.step_path} imported but no faces (2-D entities) were "
                "found -- is this a valid solid STEP export?"
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        gmsh.finalize()

    # ------------------------------------------------------------------
    def faces(self) -> list[FaceInfo]:
        """All CAD faces (2-D entities), each with its Gmsh tag ('Face ID')."""
        out = []
        for _dim, tag in gmsh.model.occ.getEntities(dim=2):
            centroid = gmsh.model.occ.getCenterOfMass(2, tag)
            bbox = gmsh.model.getBoundingBox(2, tag)
            area = gmsh.model.occ.getMass(2, tag)  # "mass" of a 2-D entity == area
            out.append(FaceInfo(tag=tag, area=area, centroid=centroid, bbox=bbox))
        return sorted(out, key=lambda f: f.tag)

    def face_tags(self) -> list[int]:
        return [f.tag for f in self.faces()]

    def print_faces(self) -> None:
        faces = self.faces()
        print(f"{self.step_path.name}: {len(faces)} faces\n")
        print(f"{'Face ID':>8}  {'area':>12}  {'centroid (x, y, z)':>28}")
        for f in faces:
            cx, cy, cz = f.centroid
            print(f"{f.tag:>8}  {f.area:12.6g}  "
                  f"({cx:8.4g}, {cy:8.4g}, {cz:8.4g})")

    def show(self, label_faces: bool = True) -> None:
        """Open the interactive Gmsh viewer. Rotate/zoom to read off Face IDs.

        This blocks until the viewer window is closed.
        """
        if label_faces:
            gmsh.option.setNumber("Geometry.SurfaceNumbers", 1)
        gmsh.fltk.run()
