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
# distinct from the red/green/blue node spheres the arrows sit among
GROUP_ARROW_COLORS = {"input": "orange", "output": "magenta"}


def _add_group_arrows(p: pv.Plotter, grid: VoxelGrid, groups,
                      u: np.ndarray | None = None, scale: float = 1.0) -> None:
    """One arrow per INPUT/OUTPUT group, on the top edge of its faces.

    Inputs point the way they are driven; outputs point the way their force
    is counted as positive. This is the picture to check directions against:
    for a gripper, the two output arrows should point at each other.

    Anchored on the top edge rather than the middle of the face because the
    isometric camera looks down from above: an arrow at mid-depth of a wall
    inside a slot sits behind the opposite wall's node spheres, while one on
    the top edge is always in view.
    """
    span = np.ptp(grid.node_xyz[grid.used_node], axis=0).max()
    # short enough that two jaws pointing at each other across a narrow slot
    # don't run into one another
    length = 0.12 * span
    for g, nodes in groups or []:
        s = np.sign(g.prescribed) if g.role == "input" else g.sign
        if len(nodes) == 0 or s == 0:
            continue    # a zero-displacement input holds still: no direction
        pts = grid.node_xyz[nodes] + (scale * u[nodes] if u is not None else 0.0)
        top = pts[pts[:, 2] >= pts[:, 2].max() - 0.5 * grid.h]
        centre = top.mean(axis=0)
        vec = np.zeros(3)
        vec[g.axis] = s
        # Keep the arrow outside the material: one pointing INTO the solid
        # (a push) ends at the face, like a finger pressing; one pointing out
        # starts there. The probe goes half an element past the face -- and,
        # for a wall anchored on its top edge, half an element down, so it
        # lands in the top layer of elements, not on the part's top plane.
        probe = centre + 0.5 * grid.h * vec
        if g.axis != 2:
            probe[2] -= 0.5 * grid.h
        start = centre - length * vec if _in_solid(grid, probe) else centre
        arrow = pv.Arrow(start=start, direction=vec, scale=length,
                         tip_length=0.3, tip_radius=0.12, shaft_radius=0.045)
        p.add_mesh(arrow, color=GROUP_ARROW_COLORS[g.role],
                   label=f"{g.name} {'-' if s < 0 else '+'}{'xyz'[g.axis]}")


def _in_solid(grid: VoxelGrid, xyz: np.ndarray) -> bool:
    """Is this point inside an active element? (grid.active is [k, i, j].)"""
    i, j, k = np.floor((xyz - grid.origin) / grid.h).astype(int)
    if not (0 <= i < grid.nelx and 0 <= j < grid.nely and 0 <= k < grid.nelz):
        return False
    return bool(grid.active[k, i, j])


def show_voxel_model(grid: VoxelGrid, bc_nodes: dict[str, np.ndarray] | None = None,
                      screenshot: str | None = None, groups=None) -> None:
    """Render the voxel part semi-transparent, BC nodes as spheres, and an
    arrow per INPUT/OUTPUT group when `groups` is given.

    Only the OUTER voxel surface is drawn (edges still show the voxel size).
    Drawing every cube stacks a translucent layer per element, which buries
    anything inside the part -- a group arrow in a slot, say -- under a dozen
    layers; the outer surface is at most two.

    Pass `screenshot` to render off-screen to an image file instead of
    opening an interactive window.
    """
    skin = _hex_grid(grid.node_xyz, grid.edof).extract_surface()

    p = pv.Plotter(off_screen=screenshot is not None)
    p.add_mesh(skin, color="lightgrey", opacity=0.3, show_edges=True,
               label=f"active elements ({grid.n_active})")

    for role, idx in (bc_nodes or {}).items():
        if len(idx) == 0:
            continue
        p.add_points(grid.node_xyz[idx], color=ROLE_COLORS.get(role, "yellow"),
                      point_size=12, render_points_as_spheres=True,
                      label=f"{role} ({len(idx)} nodes)")
    _add_group_arrows(p, grid, groups)

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
                   scale: float | None = None, screenshot: str | None = None,
                   groups=None) -> None:
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
    _add_group_arrows(p, grid, groups, u=u, scale=scale)

    p.add_scalar_bar("|u| (m)")
    p.add_legend()
    p.add_axes()
    p.view_isometric()
    p.show(screenshot=screenshot)
    if screenshot:
        print(f"  wrote {screenshot}")
