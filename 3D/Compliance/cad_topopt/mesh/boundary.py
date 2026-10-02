"""
Per-group boundary conditions -> solver-level dof arrays.

config.py describes INPUT and OUTPUT as *groups*: each group is a set of CAD
faces acting along its own signed direction ("+z", "-x", ...). The solvers
don't care about faces or groups, only about dofs, so this module flattens
the groups into six arrays (DofBCs) that Stage 4 (fem/linear_check.py) and
Stage 5 (optimization/voxel_model.py) both consume. One construction path
means the linear preview and the nonlinear optimisation can never disagree
about what the boundary conditions are.

What the arrays mean
--------------------
in_dofs / in_vals   prescribed displacement at the END of the stroke. At
                    load factor t (0..1) every input dof sits at t * in_vals,
                    so all input groups ramp together, in proportion.
out_dofs / out_k    grounded output springs. A group's stiffness is split
                    evenly over its dofs, exactly like top3d_cf.py's own ks.
out_w               the output measure: Fout = out_w @ U[out_dofs]. For a
                    group with direction sign s, w = s * k per dof, so Fout
                    is the sum over groups of each group's spring force
                    measured along ITS OWN direction. Two jaws closing
                    toward each other ("-y" on the upper, "+y" on the lower)
                    therefore both count positive -- the scissor/gripper
                    case. With one group and a "+" direction this reduces
                    to top3d_cf.py's original Fout = ks * sum(U[outdofs]).

Conflicts, resolved or rejected
-------------------------------
clamp  vs input   same dof             -> error (fixed at 0 AND driven)
input  vs input   same dof, same value -> merged
input  vs input   same dof, different  -> error (driven two ways at once)
clamp  vs output  same dof             -> output spring dropped there
                                          (a Dirichlet dof carries no spring
                                          load; harmless, reported)
input  vs output  same dof             -> error (prescribed AND free)
output vs output  same dof             -> merged: stiffnesses add, weights
                                          add. This matters beyond
                                          bookkeeping: top3d_cf.py applies
                                          springs with `Fint[outdofs] +=`,
                                          and numpy's fancy-index += applies
                                          a repeated index only ONCE, so
                                          duplicates must never reach it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from cad.face_manager import BCGroup, bc_groups, resolve
from cad.import_cad import CadModel
from mesh.face_mapping import map_faces_to_nodes, map_groups_to_nodes
from mesh.voxelize import VoxelGrid, voxelize

_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass
class DofBCs:
    fixed: np.ndarray       # clamped dofs
    in_dofs: np.ndarray     # prescribed dofs (unique, sorted)
    in_vals: np.ndarray     # full-stroke displacement at each, m
    out_dofs: np.ndarray    # output spring dofs (unique, sorted)
    out_k: np.ndarray       # spring stiffness at each, N/m
    out_w: np.ndarray       # Fout weight at each, N/m (signed)
    # (group, node ids) for every input then output group -- reporting only
    groups: list[tuple[BCGroup, np.ndarray]] = field(default_factory=list)


def _check_no_overlap(name_a, dofs_a, name_b, dofs_b):
    shared = np.intersect1d(dofs_a, dofs_b)
    if shared.size:
        raise ValueError(
            f"{shared.size} dof(s) are claimed by both '{name_a}' and "
            f"'{name_b}' -- a dof cannot be both prescribed/fixed and loaded. "
            f"Move the face selection in config.py apart, give the groups "
            f"different directions, or raise VOXEL_NEL_LONG_AXIS so the "
            f"shared-edge boundary layer thins out.")


def build_dof_bcs(clamp_nodes: np.ndarray, clamp: dict,
                  input_groups: list[BCGroup], input_nodes: list[np.ndarray],
                  output_groups: list[BCGroup], output_nodes: list[np.ndarray],
                  verbose: bool = True) -> DofBCs:
    for g, nodes in zip(input_groups + output_groups, input_nodes + output_nodes):
        if len(nodes) == 0:
            raise ValueError(
                f"{g.name} (faces {list(g.faces)}) matched no mesh nodes -- "
                f"the face may be too small for this grid; raise "
                f"VOXEL_NEL_LONG_AXIS or pick a different face.")

    clamp_axes = [_AXIS[d] for d in clamp["dofs"]]
    fixed = np.unique((3 * clamp_nodes[:, None] + np.array(clamp_axes)).ravel())

    # --- inputs --------------------------------------------------------------
    d = np.concatenate([3 * n + g.axis for g, n in zip(input_groups, input_nodes)])
    v = np.concatenate([np.full(n.size, g.prescribed)
                        for g, n in zip(input_groups, input_nodes)])
    order = np.argsort(d, kind="stable")
    d, v = d[order], v[order]
    in_dofs, start = np.unique(d, return_index=True)
    vmax = np.maximum.reduceat(v, start)
    vmin = np.minimum.reduceat(v, start)
    clash = ~np.isclose(vmax, vmin, rtol=1e-12, atol=0.0)
    if clash.any():
        raise ValueError(
            f"{int(clash.sum())} dof(s) are driven by two input groups with "
            f"different displacements (they share an edge and act along the "
            f"same axis). Give those groups the same displacement, a "
            f"different axis, or faces that do not touch.")
    in_vals = v[start]
    _check_no_overlap("clamp", fixed, "input", in_dofs)

    # --- outputs -------------------------------------------------------------
    d, k, w = [], [], []
    for g, n in zip(output_groups, output_nodes):
        dofs = 3 * n + g.axis
        # split over the group's ORIGINAL dof count, so dropping clamped
        # edge dofs below doesn't quietly stiffen what's left of the patch
        kper = g.amount / dofs.size
        d.append(dofs)
        k.append(np.full(dofs.size, kper))
        w.append(np.full(dofs.size, g.sign * kper))
    d, k, w = np.concatenate(d), np.concatenate(k), np.concatenate(w)

    on_clamp = np.isin(d, fixed)
    if on_clamp.any() and verbose:
        print(f"  note: {int(on_clamp.sum())} output dof(s) are also clamped; "
              f"dropping them from the output spring (harmless -- a "
              f"Dirichlet dof carries no spring contribution anyway).")
    d, k, w = d[~on_clamp], k[~on_clamp], w[~on_clamp]
    if d.size == 0:
        raise ValueError("every output dof is clamped -- the output faces "
                         "sit entirely on the CLAMP. Pick other faces.")

    out_dofs, inv, counts = np.unique(d, return_inverse=True, return_counts=True)
    out_k = np.bincount(inv, weights=k, minlength=out_dofs.size)
    out_w = np.bincount(inv, weights=w, minlength=out_dofs.size)
    if (counts > 1).any() and verbose:
        print(f"  note: {int((counts > 1).sum())} output dof(s) belong to more "
              f"than one output group; their springs and weights are added.")
    _check_no_overlap("input", in_dofs, "output", out_dofs)

    groups = list(zip(input_groups, input_nodes)) + list(zip(output_groups, output_nodes))
    return DofBCs(fixed=fixed, in_dofs=in_dofs, in_vals=in_vals,
                  out_dofs=out_dofs, out_k=out_k, out_w=out_w, groups=groups)


def describe(bcs: DofBCs) -> None:
    """Print one line per group: what it is, which way, how many nodes."""
    for g, nodes in bcs.groups:
        if g.role == "input":
            what = f"moves {g.prescribed * 1e3:+.4g} mm along {'xyz'[g.axis]}"
        else:
            what = f"spring {g.amount:.4g} N/m, force counted along {g.direction}"
        print(f"    {g.name:12s} faces {str(list(g.faces)):10s} "
              f"{what:44s} {len(nodes):5d} nodes")


_CONFIG = object()   # "take this argument from config.py"


def prepare(step_file=_CONFIG, clamp=_CONFIG, input_=_CONFIG, output=_CONFIG,
            design_faces=_CONFIG, nel=_CONFIG, pad=_CONFIG, verbose=True,
            strict=True):
    """The shared Stage 1-3 setup every entry point needs.

    Every argument defaults to config.py; override any of them (the
    self-test uses its own part and groups; design_faces=None means "every
    face not claimed by a BC"). Returns (grid, bc_nodes, assignment, bcs):
    bc_nodes are the per-role node unions (for viewing / structural checks),
    bcs the dof-level boundary conditions the solvers consume.

    strict=False turns a dof-level BC conflict into a printed error and
    bcs=None instead of an exception, so Stage 3 can still show you the
    mesh you need to look at to fix it. The solvers always use strict.
    """
    def pick(value, default):
        return default if value is _CONFIG else value

    step_file = pick(step_file, config.STEP_FILE)
    clamp = pick(clamp, config.CLAMP)
    input_ = pick(input_, config.INPUT)
    output = pick(output, config.OUTPUT)
    design_faces = pick(design_faces, config.DESIGN_FACES)
    nel = pick(nel, config.VOXEL_NEL_LONG_AXIS)
    pad = pick(pad, config.VOXEL_PAD_ELEMENTS)

    in_groups = bc_groups("input", input_)
    out_groups = bc_groups("output", output)
    with CadModel(step_file) as cad:
        assignment = resolve(cad.face_tags(), clamp, input_, output, design_faces)
        grid: VoxelGrid = voxelize(cad, nel_long_axis=nel, pad_elements=pad)
        cache: dict = {}
        bc_nodes = map_faces_to_nodes(assignment, grid, cache=cache)
        in_nodes = map_groups_to_nodes(in_groups, grid, cache=cache)
        out_nodes = map_groups_to_nodes(out_groups, grid, cache=cache)

    try:
        bcs = build_dof_bcs(bc_nodes["clamp"], clamp, in_groups, in_nodes,
                            out_groups, out_nodes, verbose=verbose)
    except ValueError as e:
        if strict:
            raise
        print(f"\n  BOUNDARY CONDITION ERROR: {e}")
        bcs = None
    return grid, bc_nodes, assignment, bcs
