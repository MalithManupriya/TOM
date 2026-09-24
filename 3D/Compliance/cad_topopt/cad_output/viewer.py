"""
PyVista view of the Stage 7 reconstructed surface.

Separate from mesh/viewer.py (which draws VoxelGrid cubes) because this is
looking at the *reconstructed boundary* -- the thing that actually gets
exported -- so what matters here is whether the surface is smooth, closed
and recognisably the optimised part, not where individual elements sit.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
import trimesh


def show_surface(mesh: trimesh.Trimesh, screenshot: str | None = None) -> None:
    """Render the reconstructed surface.

    Pass `screenshot` to render off-screen to an image file instead of
    opening an interactive window.
    """
    faces = np.hstack([np.full((len(mesh.faces), 1), 3, dtype=np.int64),
                       np.asarray(mesh.faces, dtype=np.int64)]).ravel()
    surf = pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)

    p = pv.Plotter(off_screen=screenshot is not None)
    p.add_mesh(surf, color="lightsteelblue", smooth_shading=True,
               show_edges=False, label=f"surface ({len(mesh.faces)} triangles)")
    p.add_legend()
    p.add_axes()
    p.show(screenshot=screenshot)
    if screenshot:
        print(f"  wrote {screenshot}")
