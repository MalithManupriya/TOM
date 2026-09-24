#!/usr/bin/env python3
"""
Stage 7/8, standalone: optimised density field -> STL -> STEP.

    python reconstruct_main.py result.npz --stl design.stl
    python reconstruct_main.py result.npz --stl design.stl --step design.step
    python reconstruct_main.py result.npz --stl d.stl --sigma 1.2 --decimate 3000
    python reconstruct_main.py result.npz --show          preview the surface

`result.npz` comes from `optimize_main.py --run --save-result result.npz`.
Reconstruction is deliberately split off from the optimisation so the
smoothing/threshold/decimation knobs below can be re-tuned in seconds
instead of re-running a multi-hour optimisation to try a different sigma.

The three knobs that matter:

    --threshold   density level counted as material (default 0.5). Lower
                  keeps more of the grey transition material, raise it for
                  a leaner part.
    --sigma       Gaussian blur, in elements, applied before thresholding
                  (default 0.8). This is what removes the voxel staircase.
                  0 gives the raw blocky surface; above ~1.5 starts eating
                  thin members.
    --decimate    target triangle count (default 5000). Stage 8 emits one
                  CAD face per triangle at roughly 2.4 kB each, so this
                  decides whether the STEP opens comfortably.

What Stage 8 produces is a faceted B-Rep solid: a real, valid solid body in
SolidWorks, but not a parametric feature tree. See cad_output/step_export.py.
"""
from __future__ import annotations

import argparse

from cad_output import isosurface, step_export
from optimization.export import load_result


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="the .npz from optimize_main.py --save-result")
    ap.add_argument("--stl", metavar="FILE", help="write the surface as STL")
    ap.add_argument("--step", metavar="FILE", help="write a STEP solid")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="density level taken as material (default 0.5)")
    ap.add_argument("--sigma", type=float, default=0.8,
                    help="pre-threshold blur, in elements (default 0.8)")
    ap.add_argument("--smooth", type=int, default=12, metavar="N",
                    help="Taubin smoothing passes on the surface (default 12)")
    ap.add_argument("--decimate", type=int, default=5000, metavar="N",
                    help="target triangle count; 0 = no decimation "
                         "(default 5000)")
    ap.add_argument("--show", action="store_true", help="3-D view of the surface")
    ap.add_argument("--screenshot", metavar="FILE",
                    help="render the view to an image file instead")
    args = ap.parse_args()

    xPhys, grid = load_result(args.result)
    print(f"  loaded {args.result}: {xPhys.size} elements on a "
          f"{grid.nelx}x{grid.nely}x{grid.nelz} grid, h = {grid.h:.6g}")

    mesh = isosurface.build(xPhys, grid, threshold=args.threshold,
                            sigma=args.sigma, smooth=args.smooth,
                            decimate=args.decimate or None)
    print()

    ok = isosurface.report(mesh, xPhys, grid, threshold=args.threshold)

    if args.stl:
        isosurface.write_stl(mesh, args.stl)

    if args.step:
        if not ok:
            raise SystemExit(
                "\nsurface checks failed -- not writing a STEP from a surface "
                "that is not a clean closed solid. Adjust --sigma/--decimate "
                "and re-run.")
        print("\nStage 8: sewing an OpenCASCADE solid")
        step_export.mesh_to_step(mesh, args.step)
        print()
        step_export.verify_step(args.step, mesh)

    if args.show or args.screenshot:
        from cad_output.viewer import show_surface
        show_surface(mesh, screenshot=args.screenshot)

    if not (args.stl or args.step or args.show or args.screenshot):
        print("\n  (nothing written -- pass --stl and/or --step)")


if __name__ == "__main__":
    main()
