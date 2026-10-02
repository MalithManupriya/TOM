#!/usr/bin/env python3
"""
Stage 4, standalone: a linear-elastic solve on the voxel mesh, to catch a
mesh/BC wiring bug before it reaches the nonlinear optimizer.

    python linear_check_main.py                 solve and report
    python linear_check_main.py --show           ... plus a deformed-shape view
    python linear_check_main.py --nel 40         override the grid resolution

The finite-element work here (shape functions, quadrature, assembly, sparse
solve) is all scikit-fem -- a real, independent solver, not top3d_cf.py's
own routine. Using a different solver for this check is deliberate: if the
mesh or the CAD-face-to-node mapping is wrong, an independent solver is much
less likely to coincidentally "confirm" the same bug than reusing the same
code path would be.

Boundary conditions are exactly Stage 5's (same mesh/boundary.py code):
    CLAMP   fixed on the clamped components
    INPUT   every input group driven along its own direction
    OUTPUT  every output group on a grounded spring along its own axis
so a look at this deformed shape is a real preview of how load will flow
through the part once Stage 5 runs the actual nonlinear optimization.

Read the per-group lines to check your directions: an output group whose
displacement comes out negative is moving AGAINST the direction you gave it
(on the solid, undesigned part), and a negative Fout means auto-Ftar will
optimise for a force in the opposite direction to the one you configured.
"""
from __future__ import annotations

import argparse

import numpy as np

import config
from fem.linear_check import run
from mesh.boundary import describe, prepare


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nel", type=int, default=config.VOXEL_NEL_LONG_AXIS, metavar="N")
    ap.add_argument("--pad", type=int, default=config.VOXEL_PAD_ELEMENTS, metavar="N")
    ap.add_argument("--show", action="store_true", help="deformed-shape 3-D view")
    ap.add_argument("--screenshot", metavar="FILE")
    ap.add_argument("--scale", type=float, default=None,
                     help="displacement scale factor for the view "
                          "(default: auto, sized to be visible)")
    args = ap.parse_args()

    try:
        grid, bc_nodes, _assignment, bcs = prepare(nel=args.nel, pad=args.pad)
    except ValueError as e:
        raise SystemExit(f"boundary conditions: {e}")
    print("\n  boundary conditions:")
    describe(bcs)

    result = run(grid, bcs, **config.MATERIAL)

    react_clamp = result.reaction[bc_nodes["clamp"]].sum(axis=0)
    react_input = result.reaction[bc_nodes["input"]].sum(axis=0)
    react_free_max = np.abs(np.delete(
        result.reaction, np.concatenate([bc_nodes["clamp"], bc_nodes["input"]]),
        axis=0)).max()

    print(f"\n  max |displacement|         {result.max_disp:.6g} m")
    print(f"  reaction sum,  clamp       {react_clamp}")
    print(f"  reaction sum,  input       {react_input}")
    print(f"  clamp + input reactions balance: "
          f"{np.allclose(react_clamp + react_input, 0, atol=1e-3 * max(1.0, np.abs(react_input).max()))}")
    print(f"  max residual on free dofs  {react_free_max:.3e}  "
          f"(should be ~0 -- equilibrium)")

    print("\n  mean displacement of each group, along its own direction:")
    for g, disp in result.group_disp:
        flag = ""
        if g.role == "output" and disp <= 0:
            flag = "   <- moves AGAINST its configured direction"
        print(f"    {g.name:12s} {g.direction:>3s}  {disp:+.4e} m{flag}")

    print(f"\n  Fout (linear preview)      {result.fout:+.4e} N")
    if result.fout == 0:
        print("\n  WARNING: zero output force -- input and output are not "
              "mechanically coupled through this geometry.")
    elif result.fout < 0 and config.OPTIMIZATION.get("Ftar") is None:
        print("\n  WARNING: Fout is negative, so the outputs as a whole move "
              "against their\n  configured directions. With Ftar = None, "
              "Stage 5 takes its target from\n  this starting response and "
              "will optimise a constant force in the OPPOSITE\n  direction. "
              "Flip the output direction(s) if they are wrong, or set\n  "
              "OPTIMIZATION['Ftar'] to a positive value to demand the "
              "configured direction.")

    if args.show or args.screenshot:
        from mesh.viewer import show_deformed
        show_deformed(grid, result.u, bc_nodes, scale=args.scale,
                      screenshot=args.screenshot, groups=bcs.groups)


if __name__ == "__main__":
    main()
