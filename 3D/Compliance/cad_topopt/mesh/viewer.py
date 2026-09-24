"""
PyVista visualization of a voxelized mesh and its face-derived BC node sets.

This is Stage 3's verification step: does the voxel model actually resemble
the CAD part, and did clamp/input/output land where you picked them in
Stage 2? Separate from cad/viewer.py (Gmsh's own GUI), because this is
looking at a derived *mesh* object (VoxelGrid), not the CAD geometry itself.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from mesh.voxelize import VoxelGrid

ROLE_COLORS = {"clamp": "red", "input": "green", "output": "blue"}


def show_voxel_model(grid: VoxelGrid, bc_nodes: dict[str, np.ndarray] | None = None,
                      screenshot: str | None = None) -> None:
    """Render active elements as semi-transparent cubes, BC nodes as spheres.

    Pass `screenshot` to render off-screen to an image file instead of
    opening an interactive window.
    """
    k_idx, i_idx, j_idx = np.nonzero(grid.active)
    centres = grid.origin + (np.stack([i_idx, j_idx, k_idx], axis=-1) + 0.5) * grid.h

    cubes = pv.PolyData(centres).glyph(
        geom=pv.Cube(x_length=grid.h, y_length=grid.h, z_length=grid.h),
        scale=False, orient=False)

    p = pv.Plotter(off_screen=screenshot is not None)
    p.add_mesh(cubes, color="lightgrey", opacity=0.3, show_edges=True,
               label=f"active elements ({grid.n_active})")

    for role, idx in (bc_nodes or {}).items():
        if len(idx) == 0:
            continue
        p.add_points(grid.node_xyz[idx], color=ROLE_COLORS.get(role, "yellow"),
                      point_size=12, render_points_as_spheres=True,
                      label=f"{role} ({len(idx)} nodes)")

    p.add_legend()
    p.add_axes()
    p.view_isometric()
    p.show(screenshot=screenshot)
    if screenshot:
        print(f"  wrote {screenshot}")


def _hex_grid(node_xyz: np.ndarray, edof: np.ndarray) -> pv.UnstructuredGrid:
    """edof's corner order is already VTK_HEXAHEDRON order (verified against
    top3d_cf.py's own reference cube in mesh/voxelize.py), so no reordering
    is needed here."""
    cells = np.hstack([np.full((edof.shape[0], 1), 8), edof]).ravel()
    return pv.UnstructuredGrid(cells, [pv.CellType.HEXAHEDRON] * edof.shape[0], node_xyz)


def show_deformed(grid: VoxelGrid, u: np.ndarray, bc_nodes: dict[str, np.ndarray] | None = None,
                   scale: float | None = None, screenshot: str | None = None) -> None:
    """Stage 4: render the mesh warped by a solved displacement field u
    (grid.node_xyz-shaped, i.e. one row per background-lattice node),
    coloured by displacement magnitude, with BC node sets marked.
    """
    mag = np.linalg.norm(u, axis=1)
    if scale is None:
        span = np.ptp(grid.node_xyz[grid.used_node], axis=0).max()
        scale = 0.1 * span / max(mag.max(), 1e-30)

    mesh = _hex_grid(grid.node_xyz + scale * u, grid.edof)
    mesh.point_data["|u|"] = mag

    p = pv.Plotter(off_screen=screenshot is not None)
    p.add_mesh(mesh, scalars="|u|", cmap="viridis", show_edges=True,
               label=f"deformed (x{scale:.3g})")

    undeformed = _hex_grid(grid.node_xyz, grid.edof)
    p.add_mesh(undeformed, color="lightgrey", opacity=0.15, style="wireframe")

    for role, idx in (bc_nodes or {}).items():
        if len(idx) == 0:
            continue
        p.add_points(grid.node_xyz[idx] + scale * u[idx], color=ROLE_COLORS.get(role, "yellow"),
                      point_size=10, render_points_as_spheres=True,
                      label=f"{role} ({len(idx)} nodes)")

    p.add_scalar_bar("|u| (m)")
    p.add_legend()
    p.add_axes()
    p.view_isometric()
    p.show(screenshot=screenshot)
    if screenshot:
        print(f"  wrote {screenshot}")
