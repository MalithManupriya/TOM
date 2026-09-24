"""
Stage 8: watertight triangle mesh -> OpenCASCADE B-Rep solid -> STEP.

Why it is done this way
-----------------------
The obvious Gmsh route -- `mesh.classifySurfaces()` + `mesh.createGeometry()`,
the reverse-engineering recipe from Gmsh's own STL remeshing tutorial -- does
NOT work for a STEP export, and this was tested rather than assumed. Those
calls build Gmsh *discrete* geometry (a reparametrised mesh), and asking for
a STEP afterwards fails outright:

    Error : No suitable CAD data found for STEP export

STEP export needs entities in the OpenCASCADE kernel. So this module builds
the solid in OCC directly: every mesh vertex becomes an OCC point, every
mesh edge an OCC line (shared between the two triangles that own it), every
triangle a planar face, and the whole lot is sewn into a closed shell and
made a volume. That produces genuine CAD data, and `gmsh.write("x.step")`
then emits a valid solid -- verified by re-importing the STEP and comparing
its volume against the source mesh (see `verify_step`).

What you get
------------
A *faceted* B-Rep: one planar CAD face per triangle, sewn into a single
solid. It opens in SolidWorks as a real solid body -- measurable, meshable,
usable for CAM -- but it is not a parametric feature tree with fillets and
extrudes, and it never will be: an organic topology-optimised shape has no
such feature history to recover. Reconstructing one is a manual remodelling
job that no automated step here or in any commercial tool (nTopology,
Inspire, Fusion) does for you.

Cost scales with triangle count, so decimate in Stage 7 first:

    triangles      build time      STEP size
        1 280          1.5 s          3.0 MB
        5 120          7.9 s         12.3 MB

(measured on this machine; roughly 2.4 kB and 1.5 ms per triangle)
"""
from __future__ import annotations

import os

import gmsh
import numpy as np
import trimesh


def _build_occ_solid(mesh: trimesh.Trimesh, verbose: bool = True) -> None:
    """Create points/lines/planar faces for every triangle, sew, make solid.

    Assumes gmsh is already initialised and the current model is empty.
    """
    occ = gmsh.model.occ
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    points = [occ.addPoint(*v) for v in verts]

    # One OCC line per undirected mesh edge, keyed (low, high) so the two
    # triangles sharing an edge reference the *same* line -- that shared
    # topology is what lets the shell sew closed instead of coming out as a
    # pile of disconnected faces.
    lines: dict[tuple[int, int], int] = {}

    def line_for(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        tag = lines.get(key)
        if tag is None:
            tag = occ.addLine(points[key[0]], points[key[1]])
            lines[key] = tag
        # the stored line runs low -> high; walking the other way is -tag
        return tag if a < b else -tag

    surfaces = []
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        loop = occ.addCurveLoop([line_for(a, b), line_for(b, c), line_for(c, a)])
        surfaces.append(occ.addPlaneSurface([loop]))

    if verbose:
        print(f"  built {len(points)} points, {len(lines)} edges, "
              f"{len(surfaces)} faces")

    shell = occ.addSurfaceLoop(surfaces, sewing=True)
    occ.addVolume([shell])
    occ.synchronize()


def mesh_to_step(mesh: trimesh.Trimesh, path: str, verbose: bool = True) -> None:
    """Write `mesh` as a STEP solid. The mesh must be watertight."""
    if not mesh.is_watertight:
        raise ValueError(
            "the mesh is not watertight, so it cannot be sewn into a closed "
            "solid. Run Stage 7's report first and fix the surface (usually: "
            "lower --decimate, or raise --sigma) before exporting a STEP.")

    # Gmsh is a native Windows binary here and reports an unwritable path only
    # after the (slow) solid has already been built, as a bare "Could not
    # create file". Checking up front costs nothing and fails immediately --
    # and catches MSYS-style paths like /tmp/x.step, which it cannot resolve.
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise ValueError(f"cannot write {path}: the directory {parent} does "
                         f"not exist.")

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("topopt_result")
        _build_occ_solid(mesh, verbose=verbose)

        n_solids = len(gmsh.model.getEntities(3))
        if n_solids != 1:
            print(f"  WARNING: sewing produced {n_solids} solid(s), expected 1. "
                  f"The STEP may be an open shell rather than a closed solid.")

        gmsh.write(path)
    finally:
        gmsh.finalize()

    print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB, "
          f"{len(mesh.faces)} CAD faces)")


def verify_step(path: str, mesh: trimesh.Trimesh) -> bool:
    """Re-import the written STEP and check it is one solid of the right size.

    Re-importing is the only honest check: it exercises OpenCASCADE's own
    reader, the same way SolidWorks will, rather than trusting that what was
    written matches what was built.
    """
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        solids = gmsh.model.getEntities(3)
        volume = sum(gmsh.model.occ.getMass(3, tag) for _dim, tag in solids)
        n_faces = len(gmsh.model.getEntities(2))
    finally:
        gmsh.finalize()

    ok = True

    one = len(solids) == 1
    ok &= one
    print(f"1. re-imports as a single solid     {one}  ({len(solids)} solid(s))")

    err = abs(volume - mesh.volume) / mesh.volume if mesh.volume else float("nan")
    matches = err < 1e-6
    ok &= matches
    print(f"2. volume matches the source mesh   {matches}  "
          f"(STEP {volume:.6g} vs mesh {mesh.volume:.6g}, rel err {err:.2e})")

    print(f"\n   CAD faces {n_faces}")
    print("\nPASS" if ok else "\nFAIL")
    return ok
