#!/usr/bin/env python3
"""
Stage 3, standalone: CAD solid -> voxel mesh -> boundary-condition node sets.

Reads the STEP file and the face selection you verified in Stage 2 (both
from config.py), rasterizes the solid into a regular grid of cubic H8
elements, and converts the CLAMP/INPUT/OUTPUT CAD faces into node sets on
that grid -- no node numbers typed by hand.

    python voxel_main.py                    build and report
    python voxel_main.py --show             ... plus a 3-D view
    python voxel_main.py --verify           ... plus structural mesh checks
    python voxel_main.py --nel 40           override the grid resolution
    python voxel_main.py --save mesh.npz    ... and write the mesh out
    python voxel_main.py --screenshot v.png render the 3-D view to a file

Cubic elements are the whole point: top3d_cf.py's neo-Hookean element,
Newton solver and adjoint sensitivities assume every element is an identical
cube, so voxelizing an arbitrary CAD shape keeps that verified physics code
usable as-is. This program produces the mesh; it does not run any FEA, and
it does not touch top3d_cf.py.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage

import config
from cad.face_manager import resolve
from cad.import_cad import CadModel
from mesh.face_mapping import map_faces_to_nodes
from mesh.voxelize import VoxelGrid, voxelize


def build(nel_long_axis: int, pad_elements: int):
    """Run the whole CAD -> voxel -> node-set chain in one Gmsh session."""
    with CadModel(config.STEP_FILE) as cad:
        assignment = resolve(cad.face_tags(), config.CLAMP, config.INPUT,
                              config.OUTPUT, config.DESIGN_FACES)
        grid = voxelize(cad, nel_long_axis=nel_long_axis,
                         pad_elements=pad_elements)
        bc_nodes = map_faces_to_nodes(assignment, grid)
    return grid, bc_nodes, assignment


def report(grid: VoxelGrid, bc_nodes: dict[str, np.ndarray], assignment) -> None:
    bbox_lo = grid.origin
    bbox_hi = grid.origin + np.array([grid.nelx, grid.nely, grid.nelz]) * grid.h
    filled = grid.n_active / (grid.nelx * grid.nely * grid.nelz)

    print()
    print(f"  background grid   {grid.nelx} x {grid.nely} x {grid.nelz} "
          f"= {grid.nelx * grid.nely * grid.nelz} cells")
    print(f"  element size h    {grid.h:.6g} (cubic)")
    print(f"  grid spans        "
          f"[{bbox_lo[0]:.4g}, {bbox_hi[0]:.4g}] x "
          f"[{bbox_lo[1]:.4g}, {bbox_hi[1]:.4g}] x "
          f"[{bbox_lo[2]:.4g}, {bbox_hi[2]:.4g}]")
    print(f"  active elements   {grid.n_active} ({100 * filled:.1f}% of the grid)")
    print(f"  solid volume      {grid.n_active * grid.h ** 3:.6g} "
          f"(voxel estimate)")
    print(f"  nodes in use      {grid.n_used_nodes}  "
          f"-> {3 * grid.n_used_nodes} dof")
    print(f"  surface nodes     {int(grid.surface_node.sum())}")
    print()
    print("  boundary conditions, from the CAD faces:")
    for role in ("clamp", "input", "output"):
        faces = assignment.faces_with_role(role)
        print(f"    {role:8s} faces {str(faces):10s} -> "
              f"{len(bc_nodes[role]):5d} nodes")

    for role, idx in bc_nodes.items():
        if len(idx) == 0:
            print(f"\n  WARNING: '{role}' matched no nodes at all. Its face(s) "
                  f"may be too small for this grid -- raise --nel.")


# top3d_cf.py's _precompute_element builds its reference cube from
#   xa = [-1, 1, 1, -1, -1, 1, 1, -1] etc, X = 0.5*h*(a+1), which puts the
# eight corners at these offsets (in units of h) from corner 0. Our edof must
# produce the same ordering or its element routine reads a garbled element.
_TOP3D_CORNER_ORDER = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)


def verify(grid: VoxelGrid, bc_nodes: dict[str, np.ndarray]) -> bool:
    """Structural checks on the mesh, in the spirit of top3d_cf.py --verify.

    These are cheap and catch the failures that would otherwise show up as a
    silently wrong FE model in Stage 5: garbled element connectivity, or a
    node carried into the system that no element supports (which makes the
    tangent singular).
    """
    ok = True

    corners = grid.node_xyz[grid.edof]
    rel = (corners - corners[:, :1, :]) / grid.h
    err = float(np.abs(rel - _TOP3D_CORNER_ORDER).max())
    ok &= err < 1e-10
    print(f"1. corner ordering vs top3d_cf.py   max err = {err:.3e}")

    distinct = all(len(set(e)) == 8 for e in grid.edof)
    ok &= distinct
    print(f"2. 8 distinct corners per element   {distinct}")

    used_idx = np.nonzero(grid.used_node)[0]
    consistent = set(np.unique(grid.edof)) == set(used_idx)
    ok &= consistent
    print(f"3. used_node == nodes in edof       {consistent}")
    print(f"   (a node with no element behind it would make Kt singular)")

    for role, idx in bc_nodes.items():
        good = bool(np.all(grid.used_node[idx])) and bool(np.all(grid.surface_node[idx]))
        ok &= good
        print(f"4. '{role}' nodes are used+surface   {good}")

    # A thin connecting feature (a flexure web, say) can be thinner than one
    # voxel and get silently severed by voxelization -- the two halves would
    # then be mechanically unconnected (a singular system, or two rigid
    # bodies) even though the real CAD part is one solid. scipy.ndimage.label
    # is a real connected-components algorithm (flood fill on the boolean
    # active-element array), not something hand-rolled here.
    struct6 = ndimage.generate_binary_structure(3, 1)  # face-sharing only --
    # the physically meaningful notion of "connected" for load transfer
    labels, n_components = ndimage.label(grid.active, structure=struct6)
    ok &= (n_components == 1)
    print(f"5. mesh is one connected piece      "
          f"{n_components == 1}  ({n_components} component(s))")
    if n_components > 1:
        sizes = ndimage.sum(grid.active, labels, index=range(1, n_components + 1))
        print(f"   component sizes (elements): {sorted(sizes, reverse=True)}")
        print(f"   raise VOXEL_NEL_LONG_AXIS -- a connecting feature is "
              f"likely thinner than one voxel at the current resolution.")

    active_flat_idx = np.nonzero(grid.active.ravel())[0]
    labels_of_active = labels.ravel()[active_flat_idx]
    for role in ("clamp", "input", "output"):
        idx = bc_nodes[role]
        if len(idx) == 0:
            continue
        touches = np.isin(grid.edof, idx).any(axis=1)
        comps = np.unique(labels_of_active[touches])
        same = len(comps) == 1
        ok &= same
        print(f"6. '{role}' sits on a single component   {same}  "
              f"(component(s) {list(comps)})")

    print("\nPASS" if ok else "\nFAIL")
    return ok


def save(path: str, grid: VoxelGrid, bc_nodes: dict[str, np.ndarray]) -> None:
    """Write the mesh + node sets so Stage 5 can load it without re-meshing."""
    np.savez_compressed(
        path,
        nelx=grid.nelx, nely=grid.nely, nelz=grid.nelz, h=grid.h,
        origin=grid.origin, active=grid.active, edof=grid.edof,
        node_xyz=grid.node_xyz, used_node=grid.used_node,
        surface_node=grid.surface_node,
        clamp_nodes=bc_nodes["clamp"], input_nodes=bc_nodes["input"],
        output_nodes=bc_nodes["output"])
    print(f"\n  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nel", type=int, default=config.VOXEL_NEL_LONG_AXIS,
                     metavar="N",
                     help="elements along the longest bounding-box axis "
                          f"(default {config.VOXEL_NEL_LONG_AXIS})")
    ap.add_argument("--pad", type=int, default=config.VOXEL_PAD_ELEMENTS,
                     metavar="N", help="empty grid layers around the part")
    ap.add_argument("--show", action="store_true", help="open the 3-D viewer")
    ap.add_argument("--screenshot", metavar="FILE",
                     help="render the 3-D view to an image file instead")
    ap.add_argument("--save", metavar="FILE", help="write the mesh to a .npz")
    ap.add_argument("--verify", action="store_true",
                     help="structural checks on the generated mesh")
    args = ap.parse_args()

    grid, bc_nodes, assignment = build(args.nel, args.pad)
    report(grid, bc_nodes, assignment)

    if args.verify:
        print()
        verify(grid, bc_nodes)

    if args.save:
        save(args.save, grid, bc_nodes)

    if args.show or args.screenshot:
        from mesh.viewer import show_voxel_model
        show_voxel_model(grid, bc_nodes, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
