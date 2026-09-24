"""
Stage 7: optimised density field -> smooth, watertight triangle mesh (STL).

The optimiser leaves a density per *active voxel*. Turning that into a
surface has three problems, each handled here:

1. Marching cubes needs a dense scalar field over the whole background grid,
   including the empty cells. `xPhys` only covers active elements, so it is
   scattered back into a full (nelx, nely, nelz) array with 0 everywhere the
   part never existed. (This is why Stage 7 reads `xPhys` + the VoxelGrid
   rather than the Stage-6 .vtu, which stores only the active hexahedra and
   so cannot be marched over directly.)

2. Thresholding a voxel field directly gives a staircase surface -- every
   face axis-aligned, every edge a 90-degree step. Gaussian-blurring the
   density field *before* thresholding turns those steps into a smooth
   level set, and is by far the biggest lever on final surface quality.
   It is applied to the zero-padded field, so the blur rounds off the
   outermost solid layer instead of smearing against the array edge.

3. Marching-cubes output is dense and unnecessarily fine (two triangles per
   voxel face). Taubin smoothing removes the residual faceting without the
   volume collapse plain Laplacian smoothing causes, and quadric decimation
   cuts the triangle count -- which matters a lot, because Stage 8 emits one
   CAD face per triangle.

Index-order note: VoxelGrid arrays are (nelz, nelx, nely) == [k, i, j]
(top3d_cf.py's convention). Everything here works in (nelx, nely, nelz) ==
[x, y, z], so marching-cubes vertices come out in xyz order directly.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from mesh.voxelize import VoxelGrid

# Taubin's shrink/dilate pair must satisfy 0 < 1/lamb - 1/nu < 0.1 for the
# filter to preserve volume rather than shrink the part (Vollmer et al.);
# lamb=0.5 / nu=0.52 sits just inside that window.
_TAUBIN_LAMB = 0.5
_TAUBIN_NU = 0.52


def dense_density(xPhys: np.ndarray, grid: VoxelGrid) -> np.ndarray:
    """Scatter the per-active-element densities onto the full background
    grid, as an (nelx, nely, nelz) array. Empty cells read 0."""
    if xPhys.size != grid.n_active:
        raise ValueError(
            f"xPhys has {xPhys.size} entries but the grid has "
            f"{grid.n_active} active elements -- these came from different "
            f"meshes (a different --nel, or a changed STEP file).")

    rho = np.zeros(grid.active.shape, dtype=float)      # (nelz, nelx, nely)
    # Boolean-mask assignment fills in C order, which is exactly the order
    # voxelize() used to build edof (`edof_all[active.ravel()]`), and hence
    # the order top3d_cf.py's xPhys is in.
    rho[grid.active] = xPhys
    return np.transpose(rho, (1, 2, 0))                 # -> (nelx, nely, nelz)


def extract_surface(xPhys: np.ndarray, grid: VoxelGrid, threshold: float = 0.5,
                    sigma: float = 0.8) -> trimesh.Trimesh:
    """Marching-cubes isosurface of the (blurred) density field, in metres.

    threshold : density level taken as the material boundary.
    sigma : Gaussian blur width, in elements. 0 disables blurring and gives
        the raw staircase surface. ~0.6-1.0 removes the stepping while
        keeping features; much above that starts eroding thin members.
    """
    rho = dense_density(xPhys, grid)

    # One layer of empty cells all round, so the isosurface always closes on
    # itself even if the part runs right to the edge of the grid.
    rho = np.pad(rho, 1, mode="constant", constant_values=0.0)
    if sigma > 0:
        rho = ndimage.gaussian_filter(rho, sigma=sigma, mode="constant", cval=0.0)

    if not (rho.min() < threshold < rho.max()):
        # Early in a run the SIMP/Heaviside field is still grey -- it starts
        # uniform at volfrac and only polarises towards 0/1 over many
        # iterations -- so nothing reaches 0.5 yet. Rather than just
        # refusing, work out the level that reproduces the material volume
        # the optimiser currently holds, which is the level a converged run
        # would have put near 0.5 anyway.
        suggestion = float(np.quantile(rho, 1.0 - xPhys.sum() / rho.size))
        raise ValueError(
            f"the density field never crosses {threshold} (range "
            f"{rho.min():.3g}..{rho.max():.3g}) -- nothing to extract.\n"
            f"  This normally means the run has not converged: the field is "
            f"still grey, so no element has reached full density.\n"
            f"  For a volume-matched cut of this field, use "
            f"--threshold {suggestion:.3g}. For a trustworthy shape, let the "
            f"optimisation converge and keep --threshold 0.5.")

    verts, faces, _normals, _values = measure.marching_cubes(
        rho, level=threshold, spacing=(grid.h, grid.h, grid.h))

    # `verts` are in metres measured from sample (0,0,0) of the padded array.
    # That sample is element index (-1,-1,-1), whose centre sits at
    # origin - 0.5h, so this offset puts the surface back in CAD coordinates.
    verts = verts + (grid.origin - 0.5 * grid.h)

    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def clean(mesh: trimesh.Trimesh, taubin_iterations: int = 12,
          target_faces: int | None = None) -> trimesh.Trimesh:
    """Repair, smooth and (optionally) decimate the raw isosurface.

    target_faces : decimate down to about this many triangles. Stage 8 emits
        one CAD face per triangle and a STEP runs roughly 2.4 kB per face, so
        this is the knob that decides whether the STEP is openable or a
        monster. None leaves the triangle count alone.
    """
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    if not mesh.is_watertight:
        mesh.fill_holes()
    mesh.fix_normals()

    # Smooth first, then decimate: quadric decimation preserves whatever
    # shape it is handed, so smoothing the dense mesh gives it better
    # information to preserve than decimating the staircase would.
    if taubin_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=_TAUBIN_LAMB, nu=_TAUBIN_NU,
                                        iterations=taubin_iterations)

    if target_faces is not None and len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=int(target_faces))
        mesh.merge_vertices()
        mesh.fix_normals()

    return mesh


def build(xPhys: np.ndarray, grid: VoxelGrid, threshold: float = 0.5,
          sigma: float = 0.8, smooth: int = 12,
          decimate: int | None = 5000) -> trimesh.Trimesh:
    """extract_surface + clean, with progress printed.

    The one code path both entry points use, so a surface reconstructed
    straight after an optimisation run and one rebuilt later from a saved
    .npz are guaranteed to be identical for the same settings.
    """
    mesh = extract_surface(xPhys, grid, threshold=threshold, sigma=sigma)
    print(f"  marching cubes -> {len(mesh.faces)} triangles")
    mesh = clean(mesh, taubin_iterations=smooth, target_faces=decimate or None)
    print(f"  after smoothing/decimation -> {len(mesh.faces)} triangles")
    return mesh


def report(mesh: trimesh.Trimesh, xPhys: np.ndarray, grid: VoxelGrid,
           threshold: float = 0.5) -> bool:
    """Checks on the extracted surface, in the style of voxel_main --verify.

    Returns whether the surface is *structurally* exportable: watertight,
    consistently wound, one body, outward-facing, inside the grid. Those are
    exactly the conditions Stage 8 needs to sew a solid, and nothing else --
    so they alone gate the STEP.

    Check 5 (volume fidelity) is reported but deliberately does NOT gate the
    export. It measures how much material the blur-and-threshold lost
    relative to the optimiser's own design, which is a judgement about
    whether the reconstruction is *faithful*, not about whether it is a
    valid solid. A grey, unconverged field thresholded near its mean will
    fail it badly while still producing a perfectly sewable surface.
    """
    ok = True

    watertight = bool(mesh.is_watertight)
    ok &= watertight
    print(f"1. watertight surface               {watertight}")

    winding = bool(mesh.is_winding_consistent)
    ok &= winding
    print(f"2. consistent winding               {winding}")

    single = mesh.body_count == 1
    ok &= single
    print(f"3. one connected body               {single}  "
          f"({mesh.body_count} body/bodies)")

    positive = mesh.volume > 0
    ok &= positive
    print(f"4. positive (outward) volume        {positive}")

    # Blur + threshold should conserve material to within a few percent; a
    # big gap means sigma is eroding the part rather than just rounding it.
    # Reported, but not part of `ok` -- see the docstring.
    voxel_vol = float((dense_density(xPhys, grid) > threshold).sum()) * grid.h ** 3
    ratio = mesh.volume / voxel_vol if voxel_vol > 0 else float("nan")
    close = 0.85 < ratio < 1.15
    print(f"5. volume vs thresholded voxels     {close}  "
          f"(mesh {mesh.volume:.4g} / voxels {voxel_vol:.4g} = {ratio:.3f})"
          f"{'' if close else '   <- fidelity only, does not block export'}")

    # Catches a wrong index order or origin offset immediately: the surface
    # can never lie outside the background grid it was extracted from.
    lo = grid.origin - grid.h
    hi = grid.origin + np.array([grid.nelx, grid.nely, grid.nelz]) * grid.h + grid.h
    inside = bool(np.all(mesh.bounds[0] >= lo) and np.all(mesh.bounds[1] <= hi))
    ok &= inside
    print(f"6. surface lies inside the grid     {inside}")

    print(f"\n   triangles {len(mesh.faces)}   vertices {len(mesh.vertices)}")
    print(f"   bounds    [{mesh.bounds[0][0]:.4g}, {mesh.bounds[1][0]:.4g}] x "
          f"[{mesh.bounds[0][1]:.4g}, {mesh.bounds[1][1]:.4g}] x "
          f"[{mesh.bounds[0][2]:.4g}, {mesh.bounds[1][2]:.4g}]")

    if ok and not close:
        print(f"\nPASS (structurally sound), but the reconstruction holds "
              f"{ratio:.0%} of the\ndesign's material -- check the shape "
              f"before trusting it. Lower --sigma, or\nlet the optimisation "
              f"converge so the density field is not still grey.")
    else:
        print("\nPASS" if ok else "\nFAIL")
    return ok


def write_stl(mesh: trimesh.Trimesh, path: str) -> None:
    mesh.export(path)
    print(f"  wrote {path}  ({len(mesh.faces)} triangles)")
