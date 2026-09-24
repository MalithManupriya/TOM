"""
Stage 5: plug the voxel mesh into top3d_cf.py's nonlinear constant-force
solver, without editing that file.

top3d_cf.py's Model builds its own box mesh inline in __init__ (node ids
from an (i, j, k) reshape, boundary conditions from array-slicing a face of
that box) and its `optimise()` function always constructs `Model(p)`
itself -- it has no hook to accept an already-built mesh. So instead of
copying optimise()'s ~80 lines here (which would mean re-verifying the
whole optimisation loop, not just the mesh plumbing), VoxelModel subclasses
Model and `patched_model()` below monkeypatches `top3d_cf.Model` to point at
it for the duration of one optimise() call. optimise() still does exactly
what it did before; it just constructs a VoxelModel instead of a box Model.

What's inherited from Model UNCHANGED (this is the entire point of
voxelizing instead of using a conforming/tet mesh -- see the project
analysis): element(), tangent(), newton(), step(), force_path(), and
_build_assembly_pattern() and _precompute_element(), because every active
element is still an identical axis-aligned cube.

What's overridden:
    __init__          mesh/BCs come from a VoxelGrid + CAD-face node sets,
                       not from nelx/nely/nelz box slicing
    _build_filter      top3d_cf.py's filter assumes every element in the
                       (nelz,nelx,nely) box exists; a voxelized shape is an
                       irregular subset, so neighbours are found with a
                       scipy.spatial.cKDTree radius query on active-element
                       centroids instead of array-offset slicing. The
                       filter weight formula itself (rmin - distance) is
                       copied verbatim from top3d_cf.py -- only how
                       neighbours are found changes, not the math.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

import config
from mesh.voxelize import VoxelGrid

sys.path.insert(0, str(config.TOP3D_CF_DIR))
import top3d_cf  # noqa: E402  (path must be set up first)

_AXIS = {"x": 0, "y": 1, "z": 2}


class VoxelModel(top3d_cf.Model):
    """top3d_cf.Model, with mesh/BCs from a voxelized CAD part.

    Constructed the same way Model is (from a single Params instance), but
    the grid/bc_nodes/role config it needs are smuggled onto that instance
    as extra attributes (`p.voxel_grid` etc.) by `build_params` below --
    top3d_cf.optimise() only ever calls `Model(p)`, so this is the one
    channel available without changing that function's signature.
    """

    def __init__(self, p):
        grid: VoxelGrid = p.voxel_grid
        bc_nodes: dict = p.bc_nodes
        clamp, input_, output = p.clamp_cfg, p.input_cfg, p.output_cfg

        self.p = p
        self._grid = grid
        self.nele = grid.n_active
        self.h = grid.h
        self.ndof = 3 * grid.node_xyz.shape[0]
        self.edof = 3 * grid.edof[:, :, None] + np.arange(3)
        self.edof = self.edof.reshape(grid.edof.shape[0], 24)

        clamp_axes = [_AXIS[d] for d in clamp["dofs"]]
        fixed = (3 * bc_nodes["clamp"][:, None] + np.array(clamp_axes)).ravel()

        in_axis = _AXIS[input_["direction"]]
        self.indofs = 3 * bc_nodes["input"] + in_axis

        # clamp vs input on the same dof is a genuine, unresolvable conflict
        # (that dof can't be both held at 0 and driven to a nonzero value).
        _check_no_overlap("clamp", fixed, "input", self.indofs)

        out_axis = _AXIS[output["direction"]]
        out_dofs_all = 3 * bc_nodes["output"] + out_axis
        # clamp vs output on the same dof is harmless, not a conflict: any
        # two faces of a box that meet along an edge share nodes there
        # unavoidably (this is geometry, not a mesh-resolution artifact), so
        # a clamp/output-adjacent design is normal, not a modelling error.
        # A spring at a dof that Dirichlet already pins to 0 contributes
        # nothing to the free system either way, so those dofs are simply
        # dropped from outdofs -- ks is still divided by the *original*
        # output node count, so the total spring stiffness the config asked
        # for (representing the whole contact patch) doesn't silently grow
        # just because a few of its edge nodes are also clamped.
        on_clamp = np.isin(out_dofs_all, fixed)
        if on_clamp.any():
            print(f"  note: {on_clamp.sum()} output dof(s) are also clamped; "
                  f"dropping them from the output spring (harmless -- a "
                  f"Dirichlet dof carries no spring contribution anyway).")
        self.ks = output["spring_stiffness"] / out_dofs_all.size
        self.outdofs = out_dofs_all[~on_clamp]

        # input vs output on the very same dof (same node AND same axis) is
        # a real conflict -- that dof can't be both prescribed and free.
        # Sharing a *node* on a different axis is fine and not checked here.
        _check_no_overlap("input", self.indofs, "output", self.outdofs)

        # Only dofs at nodes an active element actually touches may be
        # "free" -- unlike top3d_cf.py's own box mesh (every node is used by
        # construction), a voxelized mesh has background-lattice nodes
        # (VOXEL_PAD_ELEMENTS padding, or unreachable pockets) that no
        # element touches at all. Left in `free`, each is a real zero-
        # stiffness row/column in the tangent -- not ill-conditioning, an
        # exactly singular factorization, since nothing constrains it.
        used_dofs = (3 * np.nonzero(grid.used_node)[0][:, None] + np.arange(3)).ravel()
        self.free = np.setdiff1d(used_dofs, np.concatenate([fixed, self.indofs]))
        self.nfree = self.free.size
        self.dofmap = np.full(self.ndof, -1, dtype=np.int64)
        self.dofmap[self.free] = np.arange(self.nfree)
        self.out_fr = self.dofmap[self.outdofs]

        self._build_assembly_pattern()   # inherited, unchanged
        self._build_filter()             # overridden below
        self._precompute_element()       # inherited, unchanged

    def _build_filter(self):
        """Same linear-hat filter as top3d_cf.py (weight = rmin - distance,
        distance in units of one element edge), built by a KD-tree radius
        query instead of index-offset slicing, since active elements are an
        irregular subset of the background grid rather than all of it.
        Working directly in element-index space (no *h/ /h round trip)
        keeps distances exactly comparable to top3d_cf.py's own integer
        offsets.
        """
        grid, p = self._grid, self.p
        k_idx, i_idx, j_idx = np.nonzero(grid.active)
        centres = np.stack([i_idx, j_idx, k_idx], axis=-1).astype(float)

        tree = cKDTree(centres)
        pairs = tree.query_pairs(r=p.rmin, output_type="ndarray")
        d = np.linalg.norm(centres[pairs[:, 0]] - centres[pairs[:, 1]], axis=1)

        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(self.nele)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(self.nele)])
        vals = np.concatenate([p.rmin - d, p.rmin - d, np.full(self.nele, p.rmin)])

        self.H = sp.csr_matrix((vals, (rows, cols)), shape=(self.nele, self.nele))
        self.Hs = np.asarray(self.H.sum(axis=1)).ravel()


def _check_no_overlap(name_a, dofs_a, name_b, dofs_b):
    shared = np.intersect1d(dofs_a, dofs_b)
    if shared.size:
        raise ValueError(
            f"{shared.size} dof(s) are claimed by both '{name_a}' and "
            f"'{name_b}' -- top3d_cf.py's assembly requires clamp to be "
            f"disjoint from every loaded dof. Move the CLAMP/INPUT/OUTPUT "
            f"face selection in config.py apart, or raise VOXEL_NEL_LONG_AXIS "
            f"so the shared-edge boundary layer thins out.")


def build_params(grid: VoxelGrid, bc_nodes: dict, clamp: dict | None = None,
                  input_: dict | None = None, output: dict | None = None,
                  **param_overrides) -> "top3d_cf.Params":
    """A top3d_cf.Params with material/optimisation settings from config.py,
    plus the extra attributes VoxelModel.__init__ reads. Params is a plain
    (non-frozen, non-slotted) dataclass, so attaching extra attributes is
    safe -- top3d_cf.py's own code never looks at anything but its declared
    fields.

    clamp/input_/output default to config.py's live CLAMP/INPUT/OUTPUT, but
    can be overridden (e.g. by a self-test using a different mesh) -- uin
    and kout are always derived from whichever input_/output end up in
    effect, so the two can never silently disagree the way they would if
    uin/kout were read from config.py independently of clamp/input_/output.
    """
    clamp = clamp if clamp is not None else config.CLAMP
    input_ = input_ if input_ is not None else config.INPUT
    output = output if output is not None else config.OUTPUT

    kwargs = dict(E0=config.MATERIAL["E"], nu=config.MATERIAL["nu"],
                  uin=input_["displacement"], kout=output["spring_stiffness"],
                  **config.OPTIMIZATION)
    kwargs.update(param_overrides)
    p = top3d_cf.Params(**kwargs)
    # nelx/nely/nelz are only ever used by optimise()'s startup print (every
    # other method here reads self.h/self.nele, set directly in
    # VoxelModel.__init__) -- overwritten so that line reports the real
    # background-grid dimensions instead of Params' box-mesh defaults.
    p.nelx, p.nely, p.nelz = grid.nelx, grid.nely, grid.nelz
    p.voxel_grid = grid
    p.bc_nodes = bc_nodes
    p.clamp_cfg = clamp
    p.input_cfg = input_
    p.output_cfg = output
    return p


@contextmanager
def patched_model():
    """top3d_cf.optimise() always constructs `Model(p)` itself; this swaps
    that name for VoxelModel for the duration of the call, then restores it.
    top3d_cf.py's own source file is never modified."""
    original = top3d_cf.Model
    top3d_cf.Model = VoxelModel
    try:
        yield
    finally:
        top3d_cf.Model = original
