"""
Stage 6 (result export): write the optimised density field as an
unstructured-grid VTU via meshio.

top3d_cf.py's own export_vtk() hand-writes a regular-grid .vti and assumes
every element of the (nelx,nely,nelz) box exists -- the voxelized active set
is an irregular subset of that box, so it needs a real unstructured
export, not a tweak of that function. meshio is already a project
dependency (used for the Stage-3 face-mapping's surface mesh handling); this
just uses it for output too, instead of hand-writing VTK XML.
"""
from __future__ import annotations

import meshio
import numpy as np


def export_result(xPhys: np.ndarray, model, path: str) -> None:
    grid = model._grid
    used_idx = np.nonzero(grid.used_node)[0]
    remap = -np.ones(grid.node_xyz.shape[0], dtype=int)
    remap[used_idx] = np.arange(len(used_idx))

    points = grid.node_xyz[used_idx]
    cells = remap[grid.edof]  # VTK_HEXAHEDRON corner order (checked in mesh/voxelize.py)

    mesh = meshio.Mesh(points=points, cells=[("hexahedron", cells)],
                        cell_data={"density": [xPhys]})
    meshio.write(path, mesh)
    print(f"  wrote {path}  (open in ParaView; threshold 'density' at 0.5 "
          f"to see the optimised shape)")


def save_result(path: str, xPhys, model) -> None:
    """Persist the optimised density field plus the mesh it lives on.

    Stage 7/8 need `xPhys` *and* the VoxelGrid (the density array alone is
    meaningless without the grid that indexes it). Writing both means the
    CAD reconstruction can be re-tuned -- different smoothing, threshold,
    decimation -- without repeating a multi-hour optimisation, and without
    re-running Gmsh over the STEP file to rebuild the same grid.

    The .vtu written by export_result() deliberately cannot serve this
    purpose: it stores only the active hexahedra, and marching cubes needs
    the dense background grid including the empty cells.
    """
    g = model._grid
    np.savez_compressed(
        path,
        xPhys=np.asarray(xPhys),
        nelx=g.nelx, nely=g.nely, nelz=g.nelz, h=g.h, origin=g.origin,
        active=g.active, node_xyz=g.node_xyz, used_node=g.used_node,
        surface_node=g.surface_node, edof=g.edof)
    print(f"  wrote {path}  (density field + voxel grid)")


def load_result(path: str):
    """Inverse of save_result: returns (xPhys, VoxelGrid)."""
    from mesh.voxelize import VoxelGrid

    d = np.load(path)
    grid = VoxelGrid(
        nelx=int(d["nelx"]), nely=int(d["nely"]), nelz=int(d["nelz"]),
        h=float(d["h"]), origin=d["origin"], active=d["active"],
        node_xyz=d["node_xyz"], used_node=d["used_node"],
        surface_node=d["surface_node"], edof=d["edof"])
    return d["xPhys"], grid
