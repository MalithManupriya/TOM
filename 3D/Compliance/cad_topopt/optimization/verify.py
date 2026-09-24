"""
Finite-difference verification of VoxelModel, in the same spirit as
top3d_cf.py's own --verify: confirm internal force is the gradient of strain
energy, the tangent is the Jacobian of internal force, and the adjoint
sensitivity matches a finite difference on Fout. top3d_cf.py's verify()
can't be reused directly -- it hardcodes its own tiny synthetic Params/Model
internally with no hook for an external mesh -- so this repeats the same
three checks against a real (small) voxelized mesh instead.

Run this after wiring up VoxelModel and after any change to it. As
top3d_cf.py's own comment warns: wrong sensitivities do not crash, they
converge quietly to the wrong shape.
"""
from __future__ import annotations

import numpy as np

from .voxel_model import VoxelModel, build_params, patched_model, top3d_cf


def verify(grid, bc_nodes, clamp: dict, input_: dict, output: dict,
           seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    with patched_model():
        # top3d_cf.py's own verify() uses tol_nr=1e-13, tuned for its small
        # symmetric test box; this mesh (asymmetric, hole-cut) hits a higher
        # floating-point noise floor and 1e-13 simply never converges within
        # max_nr iterations -- confirmed by sweeping tol_nr (1e-11 and
        # looser all converge cleanly, only 1e-13 fails). 1e-10 is still
        # tight enough to not pollute the finite-difference comparisons
        # below (perturbation sizes d=1e-9, dd=1e-7).
        p = build_params(grid, bc_nodes, clamp, input_, output,
                          tol_nr=1e-10, verbose=False)
        m: VoxelModel = top3d_cf.Model(p)

        x = rng.uniform(0.3, 0.9, m.nele)
        Eabs = p.E0 * (p.Emin + x ** p.penal * (1 - p.Emin))
        dEabs = p.E0 * p.penal * x ** (p.penal - 1) * (1 - p.Emin)

        U = np.zeros(m.ndof)
        U[m.free] = rng.normal(0, 3e-5, m.nfree)
        U[m.indofs] = 0.1 * p.uin

        def total_energy(Uv):
            W, _, _ = m.element(Uv[m.edof], want_K=False, want_W=True)
            return float(W @ Eabs) + 0.5 * m.ks * float(Uv[m.outdofs] @ Uv[m.outdofs])

        Fint, _ = m.internal_force(U, Eabs)
        d = 1e-9
        num = np.zeros(m.nfree)
        for i, dof in enumerate(m.free):
            Up, Um = U.copy(), U.copy()
            Up[dof] += d
            Um[dof] -= d
            num[i] = (total_energy(Up) - total_energy(Um)) / (2 * d)
        e1 = np.abs(num - Fint[m.free]).max() / np.abs(Fint[m.free]).max()

        _, _, khat = m.element(U[m.edof])
        Kff = m.tangent(khat, Eabs).toarray()
        num2 = np.zeros((m.nfree, m.nfree))
        for i, dof in enumerate(m.free):
            Up, Um = U.copy(), U.copy()
            Up[dof] += d
            Um[dof] -= d
            num2[:, i] = (m.internal_force(Up, Eabs)[0][m.free]
                          - m.internal_force(Um, Eabs)[0][m.free]) / (2 * d)
        e2 = np.abs(num2 - Kff).max() / np.abs(Kff).max()

        m._Upath = None
        Fout, dFout, ok = m.force_path(Eabs, dEabs)
        assert ok, "load path failed during verification -- reduce uin or check the mesh"
        # dFout is only ever computed for k > nrise (force_path's need_lu is
        # False during the rise region, by top3d_cf.py's own design -- see
        # its docstring), so dFout[:, :nrise] is always exactly zero. Comparing
        # those entries against a nonzero finite difference would always
        # show ~100% error and has nothing to do with whether the adjoint is
        # right; top3d_cf.py's own verify() slices this same way.
        k0 = p.nrise
        worst = 0.0
        for e in rng.choice(m.nele, min(4, m.nele), replace=False):
            dd = 1e-7
            xp, xm = x.copy(), x.copy()
            xp[e] += dd
            xm[e] -= dd
            Ep = p.E0 * (p.Emin + xp ** p.penal * (1 - p.Emin))
            Em = p.E0 * (p.Emin + xm ** p.penal * (1 - p.Emin))
            m._Upath = None
            Fp_ = m.force_path(Ep)[0]
            m._Upath = None
            Fm_ = m.force_path(Em)[0]
            fd = (Fp_ - Fm_) / (2 * dd)
            worst = max(worst, np.abs(fd[k0:] - dFout[e, k0:]).max()
                        / np.abs(fd[k0:]).max())

    print(f"1. fint vs dW/du       rel err = {e1:.3e}")
    print(f"2. Kt   vs dfint/du    rel err = {e2:.3e}")
    print(f"3. adjoint dFout/dx    rel err = {worst:.3e}")
    return max(e1, e2, worst)
