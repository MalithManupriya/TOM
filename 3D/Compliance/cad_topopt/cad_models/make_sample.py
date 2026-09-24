#!/usr/bin/env python3
"""
Generate a tiny sample STEP file to try the pipeline on before using your
own SolidWorks export: a mounting block (24 x 12 x 6 mm, matching the
default Lx/nely/nelz proportions in top3d_cf.py) with a through-hole, sized
so it has more than just six trivial box faces.

Run once:  python cad_models/make_sample.py

Built directly in millimetres, matching a typical SolidWorks STEP export:
gmsh.write() always stamps a millimetre unit header on a STEP file, no
matter what numeric scale the geometry was actually built at -- so building
this in metre-scale numbers (0.024 etc) would silently mislabel a 24-micron
part as a 24 mm one. Building in millimetres up front keeps the file
self-consistent (cad/import_cad.py's Geometry.OCCTargetUnit="M" then
converts it to metres on import, same as any real STEP file).
"""
from pathlib import Path

import gmsh

OUT = Path(__file__).parent / "sample_bracket.step"

gmsh.initialize()
gmsh.model.add("sample_bracket")

box = gmsh.model.occ.addBox(0, 0, 0, 24, 12, 6)
hole = gmsh.model.occ.addCylinder(20, 6, -1, 0, 0, 8, 2.5)
gmsh.model.occ.cut([(3, box)], [(3, hole)])
gmsh.model.occ.synchronize()

gmsh.write(str(OUT))
gmsh.finalize()
print(f"wrote {OUT}")
