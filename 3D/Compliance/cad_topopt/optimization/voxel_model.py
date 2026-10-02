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
(optimization/mma.py does carry a copy of optimise(), to replace its design
update with MMA; it constructs `top3d_cf.Model(p)` the same way, so this
patch supplies the VoxelModel to both.)

What's inherited from Model UNCHANGED (this is the entire point of
voxelizing instead of using a conforming/tet mesh -- see the project
analysis): element(), tangent(), newton(), step(), force_path(), and
_build_assembly_pattern() and _precompute_element(), because every active
element is still an identical axis-aligned cube.

What's overridden:
    __init__          mesh/BCs come from a VoxelGrid + mesh/boundary.py's
                       DofBCs, not from nelx/nely/nelz box slicing
    step, force_path  per-face-group directions (config INPUT/OUTPUT lists).
                       top3d_cf.py drives every input dof with ONE scalar and
                       reads Fout as an unsigned sum over one axis; these two
                       copies differ from the originals in exactly three
                       marked lines, so inputs can move different ways and
                       outputs can count different directions (e.g. two
                       jaws closing on each other). newton(), which does the
                       actual solving, is still inherited unchanged -- it
                       already accepts a per-dof vector as its target.
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
from mesh.boundary import DofBCs
from mesh.voxelize import VoxelGrid

sys.path.insert(0, str(config.TOP3D_CF_DIR))
import top3d_cf  # noqa: E402  (path must be set up first)


class VoxelModel(top3d_cf.Model):
    """top3d_cf.Model, with mesh/BCs from a voxelized CAD part.

    Constructed the same way Model is (from a single Params instance), but
    the grid and DofBCs it needs are smuggled onto that instance as extra
    attributes (`p.voxel_grid`, `p.dof_bcs`) by `build_params` below --
    top3d_cf.optimise() only ever calls `Model(p)`, so this is the one
    channel available without changing that function's signature.
    """

    def __init__(self, p):
        grid: VoxelGrid = p.voxel_grid
        bcs: DofBCs = p.dof_bcs

        self.p = p
        self._grid = grid
        self.nele = grid.n_active
        self.h = grid.h
        self.ndof = 3 * grid.node_xyz.shape[0]
        self.edof = 3 * grid.edof[:, :, None] + np.arange(3)
        self.edof = self.edof.reshape(grid.edof.shape[0], 24)

        # Conflict checks (clamp/input/output sharing a dof) and merging of
        # overlapping groups already happened in mesh/boundary.py.
        self.indofs = bcs.in_dofs
        self.in_vals = bcs.in_vals          # full-stroke value per input dof
        self.outdofs = bcs.out_dofs
        # top3d_cf.py's ks is a scalar; here it is one stiffness per output
        # dof. Every inherited use is elementwise -- `Fint[outdofs] += ks *
        # U[outdofs]` in internal_force/newton, `np.full(n, ks)` in tangent
        # -- so the vector drops straight in.
        self.ks = bcs.out_k
        self.out_w = bcs.out_w              # Fout = out_w @ U[outdofs]

        # Only dofs at nodes an active element actually touches may be
        # "free" -- unlike top3d_cf.py's own box mesh (every node is used by
        # construction), a voxelized mesh has background-lattice nodes
        # (VOXEL_PAD_ELEMENTS padding, or unreachable pockets) that no
        # element touches at all. Left in `free`, each is a real zero-
        # stiffness row/column in the tangent -- not ill-conditioning, an
        # exactly singular factorization, since nothing constrains it.
        used_dofs = (3 * np.nonzero(grid.used_node)[0][:, None] + np.arange(3)).ravel()
        self.free = np.setdiff1d(used_dofs, np.concatenate([bcs.fixed, self.indofs]))
        self.nfree = self.free.size
        self.dofmap = np.full(self.ndof, -1, dtype=np.int64)
        self.dofmap[self.free] = np.arange(self.nfree)
        self.out_fr = self.dofmap[self.outdofs]

        self._build_assembly_pattern()   # inherited, unchanged
        self._build_filter()             # overridden below
        self._precompute_element()       # inherited, unchanged

    # ------------------------------------------------------------------ load
    # step() and force_path() below are top3d_cf.Model's own, line for line,
    # except where marked CHANGED. Everything they call -- newton(),
    # element(), tangent() -- is inherited untouched.

    def step(self, U, uin_val, Eabs, lu=None, need_lu=True):
        """Newton, retried with a fresh factorisation then with sub-stepping."""
        Ut, lu2, fhat, ok = self.newton(U, uin_val, Eabs, lu, need_lu)
        if ok:
            return Ut, lu2, fhat, True
        if lu is not None:                          # stale factorisation? retry
            Ut, lu2, fhat, ok = self.newton(U, uin_val, Eabs, None, need_lu)
            if ok:
                return Ut, lu2, fhat, True
        # CHANGED: was `float(U[self.indofs][0])`. With every input dof on
        # one value that was the same thing; with groups moving different
        # ways it would start every dof's sub-steps from the FIRST dof's
        # value -- a jump, exactly when sub-stepping is meant to go gently.
        u0 = U[self.indofs].copy()
        nsub = 2
        while nsub <= 8:
            Ut, ok_all = U.copy(), True
            for s in range(1, nsub + 1):
                target = u0 + (uin_val - u0) * s / nsub
                Ut, lu2, fhat, ok = self.newton(Ut, target, Eabs, None, need_lu)
                if not ok:
                    ok_all = False
                    break
            if ok_all:
                return Ut, lu2, fhat, True
            nsub *= 2
        return U, None, None, False

    def force_path(self, Eabs, dEabs=None):
        """Walk the load path. Returns Fout (nstep,), dFout (nele, nstep), ok."""
        p = self.p
        if getattr(self, "_Upath", None) is None or len(self._Upath) != p.nstep:
            self._Upath = [np.zeros(self.ndof) for _ in range(p.nstep)]
        Fout = np.zeros(p.nstep)
        dFout = np.zeros((self.nele, p.nstep)) if dEabs is not None else None
        rhs = np.zeros(self.nfree)
        # CHANGED: was `self.ks`. The adjoint right-hand side is dFout/dU,
        # and with Fout = out_w @ U[outdofs] that is out_w (== ks when
        # there is one "+" group, as in top3d_cf.py).
        rhs[self.out_fr] = self.out_w
        U, lu = np.zeros(self.ndof), None
        for k in range(1, p.nstep + 1):
            # CHANGED: was `p.uin * k / p.nstep` -- one scalar for every
            # input dof. Now each dof ramps to its own full-stroke value.
            target = self.in_vals * k / p.nstep
            need_lu = dEabs is not None and k > p.nrise
            guess = self._Upath[k - 1]              # warm start, previous call
            Uk, lu2, fhat, ok = self.step(guess, target, Eabs, lu, need_lu)
            if not ok:                              # fall back to continuation
                Uk, lu2, fhat, ok = self.step(U, target, Eabs, None, need_lu)
                if not ok:
                    return Fout, dFout, False
            U, lu = Uk, lu2
            self._Upath[k - 1] = U.copy()
            # CHANGED: was `self.ks * U[self.outdofs].sum()`. Each output
            # group's spring force, measured along its own direction.
            Fout[k - 1] = self.out_w @ U[self.outdofs]
            if need_lu:
                lam = np.zeros(self.ndof)
                lam[self.free] = lu.solve(rhs)      # reuse the LU factorisation
                dFout[:, k - 1] = -dEabs * np.einsum("ea,ea->e",
                                                     lam[self.edof], fhat)
        return Fout, dFout, True

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


def build_params(grid: VoxelGrid, bcs: DofBCs,
                 **param_overrides) -> "top3d_cf.Params":
    """A top3d_cf.Params with material/optimisation settings from config.py,
    plus the extra attributes VoxelModel.__init__ reads. Params is a plain
    (non-frozen, non-slotted) dataclass, so attaching extra attributes is
    safe -- top3d_cf.py's own code never looks at anything but its declared
    fields.
    """
    kwargs = dict(E0=config.MATERIAL["E"], nu=config.MATERIAL["nu"],
                  **config.OPTIMIZATION)
    kwargs.update(param_overrides)
    p = top3d_cf.Params(**kwargs)
    # uin/kout are read only by top3d_cf's own box Model and force_path,
    # both replaced by VoxelModel -- the per-dof values live in `bcs`. Set
    # here purely so a printed Params still describes this load case.
    p.uin = float(bcs.in_vals[np.argmax(np.abs(bcs.in_vals))])
    p.kout = float(bcs.out_k.sum())
    # nelx/nely/nelz are only ever used by optimise()'s startup print (every
    # other method here reads self.h/self.nele, set directly in
    # VoxelModel.__init__) -- overwritten so that line reports the real
    # background-grid dimensions instead of Params' box-mesh defaults.
    p.nelx, p.nely, p.nelz = grid.nelx, grid.nely, grid.nelz
    p.voxel_grid = grid
    p.dof_bcs = bcs
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
