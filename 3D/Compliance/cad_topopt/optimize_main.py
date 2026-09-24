#!/usr/bin/env python3
"""
Stage 5/6, standalone: run top3d_cf.py's nonlinear constant-force topology
optimization on the voxelized CAD part (Stages 1-3), using the CLAMP/INPUT/
OUTPUT faces from config.py.

    python optimize_main.py --verify         finite-difference self-test on
                                              a small, known-good mesh (not
                                              your CAD file) -- run this
                                              first, and after any change to
                                              optimization/voxel_model.py
    python optimize_main.py --time 5         time 5 optimisation iterations
                                              on your real part
    python optimize_main.py --run            full optimisation run
    python optimize_main.py --run --vtu design.vtu   ... and export the result
    python optimize_main.py --run --save-result r.npz --stl d.stl --step d.step
                                              ... and reconstruct CAD geometry

--save-result is worth passing on every real run: it stores the density
field together with the voxel grid, which is what Stages 7/8 need to rebuild
the STL/STEP later with different smoothing or decimation settings. Without
it, re-tuning the reconstruction means repeating the whole optimisation.
See reconstruct_main.py.

top3d_cf.py itself is never modified: this imports it, subclasses its Model
(optimization/voxel_model.py), and monkeypatches that one name for the
duration of the call so its own optimise() builds a VoxelModel instead of
its box mesh. Every other line of physics -- the neo-Hookean element, the
Newton solver, the adjoint, the SIMP/Heaviside optimiser loop -- runs
unchanged.
"""
from __future__ import annotations

import argparse

import config
from cad.face_manager import resolve
from cad.import_cad import CadModel
from mesh.face_mapping import map_faces_to_nodes
from mesh.voxelize import voxelize
from optimization.voxel_model import build_params, patched_model, top3d_cf


def _build_real_model():
    """Voxelize+map faces for whatever config.py currently points at."""
    with CadModel(config.STEP_FILE) as cad:
        assignment = resolve(cad.face_tags(), config.CLAMP, config.INPUT,
                              config.OUTPUT, config.DESIGN_FACES)
        grid = voxelize(cad, nel_long_axis=config.VOXEL_NEL_LONG_AXIS,
                         pad_elements=config.VOXEL_PAD_ELEMENTS)
        bc_nodes = map_faces_to_nodes(assignment, grid)
    for role in ("clamp", "input", "output"):
        if len(bc_nodes[role]) == 0:
            raise SystemExit(f"'{role}' matched no nodes -- fix the face "
                              f"selection (see voxel_main.py) before optimizing.")
    return grid, bc_nodes


def _self_test_model():
    """A small, fixed, known-non-degenerate mesh for --verify, independent
    of whatever config.py happens to point at right now (which may be mid
    face-selection, or have an unresolved clamp/output overlap)."""
    sample = config.PROJECT_DIR / "cad_models" / "sample_bracket.step"
    if not sample.exists():
        raise SystemExit(f"{sample} not found -- run "
                          f"`python cad_models/make_sample.py` once first.")
    with CadModel(sample) as cad:
        # input and output are on adjacent (edge-sharing) faces here, so
        # they must measure different directions -- both in "x" would put
        # them on the exact same dof at the shared edge, an unresolvable
        # conflict (not the harmless clamp/output case). See voxel_model.py.
        clamp = {"faces": [1], "dofs": ["x", "y", "z"]}
        input_ = {"faces": [6], "direction": "x", "displacement": -1e-4}
        output = {"faces": [4], "direction": "y", "spring_stiffness": 1.0e3}
        assignment = resolve(cad.face_tags(), clamp, input_, output, None)
        grid = voxelize(cad, nel_long_axis=8, pad_elements=1)
        bc_nodes = map_faces_to_nodes(assignment, grid)
    return grid, bc_nodes, clamp, input_, output


def _reconstruct(xPhys, grid, stl_path: str, step_path: str) -> None:
    """Stage 7/8 at default settings, straight off the finished run.

    Only the defaults are reachable from here. To tune threshold/sigma/
    decimation, pass --save-result and use reconstruct_main.py instead --
    that way each attempt costs seconds rather than another full
    optimisation.
    """
    from cad_output import isosurface, step_export

    print("\nStage 7: reconstructing the surface")
    mesh = isosurface.build(xPhys, grid)
    print()
    ok = isosurface.report(mesh, xPhys, grid)

    if stl_path:
        isosurface.write_stl(mesh, stl_path)

    if step_path:
        if not ok:
            print("\n  surface checks failed -- not writing a STEP from a "
                  "surface that is not a clean closed solid. Use "
                  "reconstruct_main.py to tune --sigma/--decimate.")
            return
        print("\nStage 8: sewing an OpenCASCADE solid")
        step_export.mesh_to_step(mesh, step_path)
        print()
        step_export.verify_step(step_path, mesh)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--time", type=int, default=0, metavar="N")
    ap.add_argument("--vtu", type=str, default="", metavar="FILE")
    ap.add_argument("--save-result", type=str, default="", metavar="FILE",
                    help="write density field + voxel grid to .npz, so "
                         "Stage 7/8 can be re-run without re-optimising")
    ap.add_argument("--stl", type=str, default="", metavar="FILE",
                    help="reconstruct and write an STL (Stage 7)")
    ap.add_argument("--step", type=str, default="", metavar="FILE",
                    help="reconstruct and write a STEP solid (Stage 8)")
    args = ap.parse_args()

    if args.verify:
        from optimization.verify import verify
        grid, bc_nodes, clamp, input_, output = _self_test_model()
        worst = verify(grid, bc_nodes, clamp, input_, output)
        print("\nPASS" if worst < 1e-4 else "\nFAIL")
        return

    if args.time or args.run:
        grid, bc_nodes = _build_real_model()
        with patched_model():
            p = build_params(grid, bc_nodes)
            if args.time:
                p.maxiter = args.time
            xPhys, hist, m = top3d_cf.optimise(p)
        if args.vtu:
            from optimization.export import export_result
            export_result(xPhys, m, args.vtu)
        if args.save_result:
            from optimization.export import save_result
            save_result(args.save_result, xPhys, m)
        if args.stl or args.step:
            _reconstruct(xPhys, grid, args.stl, args.step)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
