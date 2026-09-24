"""
Map the CLAMP/INPUT/OUTPUT CAD faces (picked in Stage 2) onto node sets of a
VoxelGrid (Stage 3), so you never type a node number by hand.

Method: for every *surface* node of the voxel grid (VoxelGrid.surface_node --
touched by the solid but not fully surrounded by it), measure its exact
distance to each assigned CAD face using Gmsh's own CAD geometry
(`gmsh.model.getClosestPoint`, i.e. the real trimmed surface, not a
triangulated proxy). A node belongs to a role if it is within `tol` of one of
that role's faces. Interior nodes are never tested, since they cannot be on
any face and it would only waste time.

Only Stage-1/2 objects (CadModel + FaceAssignment) and Stage-3's VoxelGrid
are needed here; nothing about the nonlinear solver is touched.
"""
from __future__ import annotations

import numpy as np
import gmsh

from cad.face_manager import FaceAssignment
from mesh.voxelize import VoxelGrid

_BC_ROLES = ("clamp", "input", "output")


def map_faces_to_nodes(assignment: FaceAssignment, grid: VoxelGrid,
                        tol: float | None = None) -> dict[str, np.ndarray]:
    """Return {"clamp": node_ids, "input": node_ids, "output": node_ids}.

    tol defaults to half an element. That is the useful compromise: on a
    face aligned with the grid it captures exactly the one layer of nodes
    lying on the face (the next layer in is a full h away, so it is
    excluded), while on a slanted or curved face -- where the voxel
    boundary staircases and no node sits exactly on the surface -- it still
    captures the nodes hugging it. A tol of a full h silently grabs a
    second layer of nodes on flat faces.
    """
    tol = 0.5 * grid.h if tol is None else tol
    cand_idx = np.nonzero(grid.surface_node)[0]
    cand_xyz = grid.node_xyz[cand_idx]

    result: dict[str, np.ndarray] = {}
    for role in _BC_ROLES:
        faces = assignment.faces_with_role(role)
        best = np.full(len(cand_idx), np.inf)
        for tag in faces:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
            in_box = ((cand_xyz[:, 0] >= xmin - tol) & (cand_xyz[:, 0] <= xmax + tol) &
                      (cand_xyz[:, 1] >= ymin - tol) & (cand_xyz[:, 1] <= ymax + tol) &
                      (cand_xyz[:, 2] >= zmin - tol) & (cand_xyz[:, 2] <= zmax + tol))
            for n in np.nonzero(in_box)[0]:
                coord, _ = gmsh.model.getClosestPoint(2, tag, cand_xyz[n].tolist())
                d = float(np.linalg.norm(np.asarray(coord) - cand_xyz[n]))
                if d < best[n]:
                    best[n] = d
        result[role] = cand_idx[best <= tol]

    for a, b in (("clamp", "input"), ("clamp", "output"), ("input", "output")):
        both = set(result[a]) & set(result[b])
        if both:
            print(f"  note: {len(both)} node(s) are in both '{a}' and '{b}'. "
                  f"That is expected where the two faces share an edge, but "
                  f"Stage 5 cannot both fix and load the same node -- pick "
                  f"faces that do not touch, or decide which role wins.")
    return result
