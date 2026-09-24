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

Boundary conditions mirror top3d_cf.py's roles directly:
    CLAMP  fixed in all 3 components
    INPUT  prescribed displacement in one component (config.INPUT)
    OUTPUT a grounded spring in one component (config.OUTPUT)
so a look at this deformed shape is a real preview of how load will flow
through the part once Stage 5 runs the actual nonlinear optimization.
"""
from __future__ import annotations

import argparse

import numpy as np

import config
from cad.face_manager import resolve
from cad.import_cad import CadModel
from fem.linear_check import run
from mesh.face_mapping import map_faces_to_nodes
from mesh.voxelize import voxelize


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

    with CadModel(config.STEP_FILE) as cad:
        assignment = resolve(cad.face_tags(), config.CLAMP, config.INPUT,
                              config.OUTPUT, config.DESIGN_FACES)
        grid = voxelize(cad, nel_long_axis=args.nel, pad_elements=args.pad)
        bc_nodes = map_faces_to_nodes(assignment, grid)

    for role in ("clamp", "input", "output"):
        if len(bc_nodes[role]) == 0:
            raise SystemExit(f"'{role}' matched no nodes -- fix the face "
                              f"selection or raise --nel before solving.")

    result = run(grid, bc_nodes, config.CLAMP, config.INPUT, config.OUTPUT,
                 **config.MATERIAL)

    react_clamp = result.reaction[bc_nodes["clamp"]].sum(axis=0)
    react_input = result.reaction[bc_nodes["input"]].sum(axis=0)
    react_free_max = np.abs(np.delete(
        result.reaction, np.concatenate([bc_nodes["clamp"], bc_nodes["input"]]),
        axis=0)).max()

    print(f"\n  max |displacement|         {result.max_disp:.6g} m")
    print(f"  mean displacement, output  {result.output_disp_mean}")
    print(f"  reaction sum,  clamp       {react_clamp}")
    print(f"  reaction sum,  input       {react_input}")
    print(f"  clamp + input reactions balance: "
          f"{np.allclose(react_clamp + react_input, 0, atol=1e-3 * max(1.0, np.abs(react_input).max()))}")
    print(f"  max residual on free dofs  {react_free_max:.3e}  "
          f"(should be ~0 -- equilibrium)")

    if result.output_disp_mean[{"x": 0, "y": 1, "z": 2}[config.OUTPUT["direction"]]] == 0:
        print("\n  WARNING: zero displacement at the output nodes in the "
              "configured direction -- input and output may not be "
              "mechanically coupled through this geometry.")

    if args.show or args.screenshot:
        from mesh.viewer import show_deformed
        show_deformed(grid, result.u, bc_nodes, scale=args.scale,
                      screenshot=args.screenshot)


if __name__ == "__main__":
    main()
