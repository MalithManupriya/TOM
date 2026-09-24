"""
Turn the imported CAD solid into a regular background grid of cubic H8
elements, keeping only the elements whose centre lies inside the solid.

Why this exists (see the project README/analysis): top3d_cf.py's nonlinear
element routine, Newton solver and adjoint sensitivities all assume every
element is an identical axis-aligned cube -- that is what lets it precompute
one shared dN/dX and reuse it everywhere. Voxelizing preserves that property
for an arbitrary CAD shape (only the *set* of elements that exist changes,
never their shape), so none of that verified physics code has to change.
An unstructured tetrahedral mesh does not have this property.

Node/element indexing mirrors top3d_cf.py's own convention exactly
(arrays shaped (nelz, nelx, nely) / (nelz+1, nelx+1, nely+1), axis order
[k, i, j] = [z, x, y]) so a future Model class can reuse this structure with
minimal changes.

Geometry-in-the-loop: a *watertight triangulated surface* of the CAD solid
is needed to test "is this point inside the part". Gmsh's own 2-D (surface)
mesher builds that triangulation directly from the exact CAD faces --
robust even on shapes that would be hard to volume-mesh -- and `trimesh`
(a mature, standard computational-geometry library) does the actual
inside/outside test. No custom geometry code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
import gmsh

# The 8 corner offsets of one H8 element, in the same order top3d_cf.py
# uses for `edof` (so a per-element quantity computed one way lines up with
# the other). Offsets are (dk, di, dj) applied to the element's own (k,i,j).
_CORNER_OFFSETS = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1),
                    (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]


@dataclass
class VoxelGrid:
    nelx: int
    nely: int
    nelz: int
    h: float                    # cubic element edge length
    origin: np.ndarray          # (3,) xyz of node index (k=0, i=0, j=0)
    active: np.ndarray          # bool (nelz, nelx, nely) -- element exists
    node_xyz: np.ndarray        # (nnode, 3) every background-lattice node
    used_node: np.ndarray       # bool (nnode,) touched by >=1 active element
    surface_node: np.ndarray    # bool (nnode,) used AND on the outer boundary
    edof: np.ndarray            # (n_active, 8) node ids, active elements only

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    @property
    def n_used_nodes(self) -> int:
        return int(self.used_node.sum())


def _surface_triangulation(mesh_size: float) -> trimesh.Trimesh:
    """Mesh every CAD face (2-D) and hand the triangulation to trimesh."""
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size / 5)
    gmsh.model.mesh.generate(2)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    verts = np.asarray(coords, dtype=float).reshape(-1, 3)
    tag_to_row = {t: i for i, t in enumerate(node_tags)}

    tris = []
    for dim, surf_tag in gmsh.model.occ.getEntities(dim=2):
        etypes, _etags, enodes = gmsh.model.mesh.getElements(dim=2, tag=surf_tag)
        for et, nodes in zip(etypes, enodes):
            props = gmsh.model.mesh.getElementProperties(et)
            nper = props[3]
            if nper != 3:
                raise RuntimeError(
                    f"surface mesh produced a non-triangular element "
                    f"(type {et}); this should not happen with Gmsh's "
                    f"default 2-D algorithm.")
            rows = np.array([tag_to_row[n] for n in nodes]).reshape(-1, 3)
            tris.append(rows)
    faces = np.concatenate(tris, axis=0)

    surf = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if not surf.is_watertight:
        raise RuntimeError(
            "the CAD boundary did not mesh into a watertight surface "
            "(there may be a gap/sliver in the STEP geometry). Voxelization "
            "needs a closed solid to classify inside/outside reliably.")
    return surf


def voxelize(cad_model, nel_long_axis: int = 24, pad_elements: int = 1) -> VoxelGrid:
    """Rasterize the currently-imported CAD solid into a VoxelGrid.

    nel_long_axis : element count along the longest bounding-box dimension;
        elements are cubes, so this sets the edge length h and the other two
        axes get however many whole cubes fit.
    pad_elements : extra empty layers of background grid around the CAD
        bounding box, so boundary faces at the very edge of the box still
        get a full ring of background nodes around them.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    h = max(Lx, Ly, Lz) / nel_long_axis

    # OpenCASCADE reports a bounding box inflated by a small gap tolerance
    # (~1e-7 m per side), so an axis that holds a whole number of elements
    # comes out as e.g. 12.0001 and a bare ceil() would add an entire
    # spurious element layer. The epsilon is in units of elements, and is
    # sized to swallow that CAD tolerance while still rounding a genuine
    # part-full cell (12.05) up.
    def n_cells(length: float) -> int:
        return max(1, int(np.ceil(length / h - 1e-3)))

    nelx = n_cells(Lx) + 2 * pad_elements
    nely = n_cells(Ly) + 2 * pad_elements
    nelz = n_cells(Lz) + 2 * pad_elements
    origin = np.array([xmin, ymin, zmin]) - pad_elements * h

    surf = _surface_triangulation(mesh_size=0.5 * h)

    # element centres, indexed [k, i, j] exactly like top3d_cf.py's eidx
    kk, ii, jj = np.meshgrid(np.arange(nelz), np.arange(nelx), np.arange(nely),
                              indexing="ij")
    centres = origin + (np.stack([ii, jj, kk], axis=-1) + 0.5) * h
    # trimesh wants (x, y, z); centres above is already ordered that way
    # because ii->x, jj->y, kk->z (see stack order).
    inside = surf.contains(centres.reshape(-1, 3))
    active = inside.reshape(nelz, nelx, nely)

    # background-lattice nodes, indexed [k, i, j], same convention as
    # top3d_cf.py's `nid`
    nk, ni, nj = np.meshgrid(np.arange(nelz + 1), np.arange(nelx + 1),
                              np.arange(nely + 1), indexing="ij")
    node_xyz = origin + np.stack([ni, nj, nk], axis=-1).reshape(-1, 3) * h
    nid = np.arange(node_xyz.shape[0]).reshape(nelz + 1, nelx + 1, nely + 1)

    corners = [nid[kk + dk, ii + di, jj + dj] for dk, di, dj in _CORNER_OFFSETS]
    edof_all = np.stack(corners, axis=-1).reshape(-1, 8)   # all elements, active or not
    edof = edof_all[active.ravel()]

    used_node = np.zeros(node_xyz.shape[0], dtype=bool)
    used_node[np.unique(edof)] = True

    touched = np.zeros((nelz + 1, nelx + 1, nely + 1), dtype=np.int8)
    for dk, di, dj in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                       (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)):
        touched[dk:dk + nelz, di:di + nelx, dj:dj + nely] += active.astype(np.int8)
    surface_node = used_node & (touched.ravel() < 8)

    return VoxelGrid(nelx=nelx, nely=nely, nelz=nelz, h=h, origin=origin,
                      active=active, node_xyz=node_xyz, used_node=used_node,
                      surface_node=surface_node, edof=edof)
