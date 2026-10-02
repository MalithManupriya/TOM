#!/usr/bin/env python3
"""
Stage 5/6, standalone: run top3d_cf.py's nonlinear constant-force topology
optimization on the voxelized CAD part (Stages 1-3), using the CLAMP/INPUT/
OUTPUT faces from config.py.

    python optimize_main.py --verify         finite-difference self-test on
                                              a small, known-good mesh (not
                                              your CAD file) -- run this
                                              first, and after any change to
                                              optimization/voxel_model.py or
                                              optimization/mma.py
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
duration of the call so the optimiser loop builds a VoxelModel instead of
its box mesh. Every line of physics -- the neo-Hookean element, the Newton
solver, the adjoint, the filter and Heaviside projection -- runs unchanged.
Only the design update is swappable (config.OPTIMIZER): "mma" runs
optimization/mma.py's copy of top3d_cf.optimise() with a GCMMA step
(Svanberg's globally convergent MMA), "pg" runs top3d_cf.optimise() itself.
"""
from __future__ import annotations

import argparse

import config
from mesh.boundary import describe, prepare
from optimization import mma
from optimization.voxel_model import build_params, patched_model, top3d_cf

OPTIMISERS = {"mma": mma.optimise, "pg": top3d_cf.optimise}


def _build_real_model():
    """Voxelize + map faces + build dof BCs for whatever config.py points at."""
    try:
        grid, _bc_nodes, _assignment, bcs = prepare()
    except ValueError as e:
        raise SystemExit(f"boundary conditions: {e}")
    print("  boundary conditions:")
    describe(bcs)
    return grid, bcs


# Two self-test load cases on the same small part, independent of whatever
# config.py happens to point at right now (which may be mid face-selection).
#
# sample_bracket.step faces:  1 x=0 end, 6 x=max end, 2 y=0, 4 y=max,
#                             3/5 top/bottom, 7 the through-hole.
_SELF_TESTS = {
    # The original case: one input, one output, one direction each. Must
    # reproduce the numbers from before per-group BCs existed -- this is
    # the regression check that the refactor changed nothing for them.
    # Input and output share an edge, so they must act on different axes.
    "single direction": dict(
        clamp={"faces": [1], "dofs": ["x", "y", "z"]},
        input_={"faces": [6], "direction": "x", "displacement": -1e-4},
        output={"faces": [4], "direction": "y", "spring_stiffness": 1.0e3},
    ),
    # Everything the per-group code adds: a signed input direction, a second
    # input on another face and axis, and two outputs with OPPOSITE signs and
    # different stiffnesses -- jaws 2 and 4 closing on each other along y.
    "multi direction": dict(
        clamp={"faces": [1], "dofs": ["x", "y", "z"]},
        input_=[{"faces": [6], "direction": "-x", "displacement": 1e-4},
                {"faces": [7], "direction": "+y", "displacement": 5e-5}],
        output=[{"faces": [2], "direction": "+y", "spring_stiffness": 400.0},
                {"faces": [4], "direction": "-y", "spring_stiffness": 1200.0}],
    ),
}


def _self_test_models():
    sample = config.PROJECT_DIR / "cad_models" / "sample_bracket.step"
    if not sample.exists():
        raise SystemExit(f"{sample} not found -- run "
                         f"`python cad_models/make_sample.py` once first.")
    for label, case in _SELF_TESTS.items():
        grid, _bc_nodes, _assignment, bcs = prepare(
            step_file=sample, design_faces=None, nel=8, pad=1, **case)
        yield label, grid, bcs


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
        worst = 0.0
        for label, grid, bcs in _self_test_models():
            print(f"\n[{label}]")
            describe(bcs)
            worst = max(worst, verify(grid, bcs))
        print("\n[GCMMA vs SLSQP]")
        worst = max(worst, mma.self_test())
        print("\nPASS" if worst < 1e-4 else "\nFAIL")
        return

    if args.time or args.run:
        if config.OPTIMIZER not in OPTIMISERS:
            raise SystemExit(f"config.OPTIMIZER = {config.OPTIMIZER!r}; "
                             f"expected one of {sorted(OPTIMISERS)}")
        grid, bcs = _build_real_model()
        with patched_model():
            p = build_params(grid, bcs)
            if args.time:
                p.maxiter = args.time
            xPhys, hist, m = OPTIMISERS[config.OPTIMIZER](p)
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
