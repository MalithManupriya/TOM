#!/usr/bin/env python3
"""
top3d_cf.py -- 3-D constant-force compliant mechanism topology optimisation.

Solves
    min_x   J = (1/|P|) sum_{k in P} [ (Fout_k - Ftar) / Ftar ]^2
    s.t.    V(x)/V0 <= volfrac
            R(U_k, x) = 0        (nonlinear equilibrium at every load step)
            0 <= x <= 1

P is the plateau window (steps nrise+1 ... nstep). Steps 1..nrise are the
preload / rise region and are left unconstrained on purpose -- constraining
them would demand a force step at u = 0.

Physics: total-Lagrangian H8 elements, compressible neo-Hookean,
displacement-controlled Newton-Raphson, adjoint sensitivities through the
converged tangent. Geometric nonlinearity is not optional: under linear
kinematics Fout is proportional to u and a plateau cannot exist.

    python3 top3d_cf.py --verify        finite-difference check of the element
                                        and the adjoint sensitivities
    python3 top3d_cf.py --run           optimise with defaults
    python3 top3d_cf.py --time 3        time 3 optimisation iterations

Author's note: run --verify after ANY edit to the element routine. Wrong
sensitivities do not crash; they converge quietly to the wrong shape.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# Voigt ordering: [11, 22, 33, 12, 23, 13]
VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


# ----------------------------------------------------------------------------
@dataclass
class Params:
    # mesh -------------------------------------------------------------------
    nelx: int = 32          # elements along x (length)
    nely: int = 12          # elements along y (height)
    nelz: int = 6           # elements along z (width)
    Lx: float = 0.024       # m, domain length; elements are cubes h = Lx/nelx
    # material ---------------------------------------------------------------
    E0: float = 2.0e9       # Pa, solid modulus (printed resin / nylon)
    Emin: float = 1e-6      # void modulus relative to E0
    nu: float = 0.40
    # optimisation -----------------------------------------------------------
    volfrac: float = 0.30
    penal: float = 3.0
    rmin: float = 2.0       # filter radius in elements
    beta: float = 1.0       # Heaviside sharpness
    beta_max: float = 16.0
    beta_iter: int = 40     # double beta every this many iterations
    eta: float = 0.5        # projection threshold
    move: float = 0.05
    maxiter: int = 100
    # load case --------------------------------------------------------------
    uin: float = -0.004     # m, TOTAL input stroke (negative = pushing in)
    nstep: int = 10         # load steps over that stroke
    nrise: int = 3          # steps 1..nrise are the rise region
    kout: float = 1.0e3     # N/m, total output spring = tissue contact stiffness
    Ftar: float | None = None   # N, target plateau force; None = auto
    # solver -----------------------------------------------------------------
    tol_nr: float = 1e-7
    max_nr: int = 25
    verbose: bool = True


# ----------------------------------------------------------------------------
class Model:
    """Mesh, boundary conditions, filter and nonlinear FE machinery."""

    def __init__(self, p: Params):
        self.p = p
        nelx, nely, nelz = p.nelx, p.nely, p.nelz
        self.nele = nelx * nely * nelz
        self.h = p.Lx / nelx
        nnode = (nelx + 1) * (nely + 1) * (nelz + 1)
        self.ndof = 3 * nnode

        # --- connectivity ----------------------------------------------------
        nid = np.arange(nnode).reshape(nelz + 1, nelx + 1, nely + 1)  # [k,i,j]
        kk, ii, jj = np.meshgrid(np.arange(nelz), np.arange(nelx),
                                 np.arange(nely), indexing="ij")
        kk, ii, jj = kk.ravel(), ii.ravel(), jj.ravel()
        corners = [nid[kk, ii, jj], nid[kk, ii + 1, jj],
                   nid[kk, ii + 1, jj + 1], nid[kk, ii, jj + 1],
                   nid[kk + 1, ii, jj], nid[kk + 1, ii + 1, jj],
                   nid[kk + 1, ii + 1, jj + 1], nid[kk + 1, ii, jj + 1]]
        self.edof = np.empty((self.nele, 24), dtype=np.int64)
        for a, n in enumerate(corners):
            self.edof[:, 3 * a:3 * a + 3] = 3 * n[:, None] + np.arange(3)

        # --- boundary conditions --------------------------------------------
        #   clamped : whole x = 0 face (the mount)
        #   input   : line (x = Lx, y = 0,  all z), prescribed x-displacement
        #   output  : line (x = Lx, y = Ly, all z), springs to ground, x-dir
        # Both ports sit on the free end and act along x, so they are strongly
        # coupled through the body: the drive pushes in at the bottom edge, the
        # jaw pushes out at the top edge. Layouts that cross directions (push
        # in x, read out in y) couple far too weakly on a uniform start -- the
        # initial force path comes out non-monotone and sign-indefinite, and
        # the optimiser has nothing to latch onto.
        clamp_n = nid[:, 0, :].ravel()
        fixed = (3 * clamp_n[:, None] + np.arange(3)).ravel()
        self.indofs = 3 * nid[:, nelx, 0] + 0
        self.outdofs = 3 * nid[:, nelx, nely] + 0
        self.ks = p.kout / self.outdofs.size          # spring per output node

        self.free = np.setdiff1d(np.arange(self.ndof),
                                 np.concatenate([fixed, self.indofs]))
        self.nfree = self.free.size
        self.dofmap = np.full(self.ndof, -1, dtype=np.int64)
        self.dofmap[self.free] = np.arange(self.nfree)
        self.out_fr = self.dofmap[self.outdofs]        # output dofs, free index

        self._build_assembly_pattern()
        self._build_filter()
        self._precompute_element()

    # ------------------------------------------------------------------ setup
    def _build_assembly_pattern(self):
        """Pre-sort the sparse pattern once so each assembly is a bincount."""
        rows = np.repeat(self.edof, 24, axis=1).ravel()   # flat c = I*24 + J
        cols = np.tile(self.edof, 24).ravel()
        fr, fc = self.dofmap[rows], self.dofmap[cols]
        self.keep = (fr >= 0) & (fc >= 0)
        fr, fc = fr[self.keep], fc[self.keep]
        # output springs live on the diagonal; append them to the same pattern
        fr = np.concatenate([fr, self.out_fr])
        fc = np.concatenate([fc, self.out_fr])
        order = np.lexsort((fc, fr))
        self.order = order
        rs, cs = fr[order], fc[order]
        first = np.ones(rs.size, bool)
        first[1:] = (rs[1:] != rs[:-1]) | (cs[1:] != cs[:-1])
        self.group = np.cumsum(first) - 1
        self.ngroup = int(self.group[-1]) + 1
        self.Kindices = cs[first]
        counts = np.bincount(rs[first], minlength=self.nfree)
        self.Kindptr = np.concatenate([[0], np.cumsum(counts)])

    def _build_filter(self):
        """Linear-hat density filter, built by offsets (no per-element loop)."""
        p = self.p
        nelx, nely, nelz = p.nelx, p.nely, p.nelz
        eidx = np.arange(self.nele).reshape(nelz, nelx, nely)
        R = int(np.ceil(p.rmin)) - 1
        rows, cols, vals = [], [], []
        for dk in range(-R, R + 1):
            for di in range(-R, R + 1):
                for dj in range(-R, R + 1):
                    d = np.sqrt(di * di + dj * dj + dk * dk)
                    if d >= p.rmin:
                        continue
                    sk = slice(max(0, -dk), nelz - max(0, dk))
                    si = slice(max(0, -di), nelx - max(0, di))
                    sj = slice(max(0, -dj), nely - max(0, dj))
                    tk = slice(max(0, dk), nelz - max(0, -dk))
                    ti = slice(max(0, di), nelx - max(0, -di))
                    tj = slice(max(0, dj), nely - max(0, -dj))
                    src = eidx[sk, si, sj].ravel()
                    tgt = eidx[tk, ti, tj].ravel()
                    rows.append(tgt)
                    cols.append(src)
                    vals.append(np.full(src.size, p.rmin - d))
        self.H = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.nele, self.nele))
        self.Hs = np.asarray(self.H.sum(axis=1)).ravel()

    def _precompute_element(self):
        """dN/dX and weight*detJ at the 8 Gauss points (identical for all elements)."""
        h = self.h
        xa = np.array([-1, 1, 1, -1, -1, 1, 1, -1], float)
        ya = np.array([-1, -1, 1, 1, -1, -1, 1, 1], float)
        za = np.array([-1, -1, -1, -1, 1, 1, 1, 1], float)
        g = 1.0 / np.sqrt(3.0)
        self.dNdX, self.wdet = [], []
        for xi in (-g, g):
            for et in (-g, g):
                for ze in (-g, g):
                    dN = 0.125 * np.array([
                        xa * (1 + et * ya) * (1 + ze * za),
                        ya * (1 + xi * xa) * (1 + ze * za),
                        za * (1 + xi * xa) * (1 + et * ya)])
                    X = 0.5 * h * np.stack([xa + 1, ya + 1, za + 1], axis=1)
                    J0 = dN @ X
                    self.dNdX.append(np.linalg.solve(J0, dN))
                    self.wdet.append(abs(np.linalg.det(J0)))   # weights are 1
        self.mu1 = 1.0 / (2.0 * (1.0 + self.p.nu))             # at E = 1
        self.lam1 = self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))

    # -------------------------------------------------------------- element
    def element(self, Ue, want_K=True, want_W=False):
        """Unit-modulus neo-Hookean H8, vectorised over all elements.

        Ue : (nele, 24)   ->  W (nele,), fhat (nele, 24), khat (nele, 24, 24)
        """
        nele = Ue.shape[0]
        mu1, lam1 = self.mu1, self.lam1
        Uc = Ue.reshape(nele, 8, 3)
        fhat = np.zeros((nele, 24))
        khat = np.zeros((nele, 24, 24)) if want_K else None
        W = np.zeros(nele) if want_W else None
        eye3 = np.eye(3)

        for dNdX, w in zip(self.dNdX, self.wdet):
            # F[e,k,j] = delta + sum_a U[e,a,k] dN[j,a]
            F = eye3 + np.einsum("eak,ja->ekj", Uc, dNdX)
            C = np.einsum("eki,ekj->eij", F, F)
            detC = np.maximum(np.linalg.det(C), 1e-8)   # guard on inverted voids
            lnJ = 0.5 * np.log(detC)
            Ci = np.linalg.inv(C)
            if want_W:
                I1 = np.trace(C, axis1=1, axis2=2)
                W += w * (0.5 * mu1 * (I1 - 3 - 2 * lnJ) + 0.5 * lam1 * lnJ ** 2)
            S = mu1 * (eye3 - Ci) + lam1 * lnJ[:, None, None] * Ci

            B = np.zeros((nele, 6, 24))
            for I, (i, j) in enumerate(VOIGT):
                for k in range(3):
                    val = F[:, k, i][:, None] * dNdX[j][None, :]
                    if i != j:
                        val = val + F[:, k, j][:, None] * dNdX[i][None, :]
                    B[:, I, k::3] = val
            Sv = np.stack([S[:, i, j] for (i, j) in VOIGT], axis=1)
            fhat += w * np.einsum("eIa,eI->ea", B, Sv)

            if not want_K:
                continue
            c = 2.0 * (mu1 - lam1 * lnJ)
            D = np.empty((nele, 6, 6))
            for I, (i, j) in enumerate(VOIGT):
                for J, (k, l) in enumerate(VOIGT):
                    D[:, I, J] = (lam1 * Ci[:, i, j] * Ci[:, k, l]
                                  + c * 0.5 * (Ci[:, i, k] * Ci[:, j, l]
                                               + Ci[:, i, l] * Ci[:, j, k]))
            khat += w * (B.transpose(0, 2, 1) @ (D @ B))       # material part
            G = np.einsum("ia,eij,jb->eab", dNdX, S, dNdX)     # geometric part
            for k in range(3):
                khat[:, k::3, k::3] += w * G
        return W, fhat, khat

    # ------------------------------------------------------------- assembly
    def internal_force(self, U, Eabs):
        _, fhat, _ = self.element(U[self.edof], want_K=False)
        Fint = np.bincount(self.edof.ravel(),
                           weights=(fhat * Eabs[:, None]).ravel(),
                           minlength=self.ndof)
        Fint[self.outdofs] += self.ks * U[self.outdofs]
        return Fint, fhat

    def tangent(self, khat, Eabs):
        vals = (khat * Eabs[:, None, None]).reshape(-1)[self.keep]
        vals = np.concatenate([vals, np.full(self.out_fr.size, self.ks)])
        data = np.bincount(self.group, weights=vals[self.order],
                           minlength=self.ngroup)
        return sp.csr_matrix((data, self.Kindices, self.Kindptr),
                             shape=(self.nfree, self.nfree))

    # --------------------------------------------------------------- solver
    # splu dominates the runtime (roughly 0.46 s vs 0.11 s for a tangent
    # assembly on the default mesh), so the solver is a MODIFIED Newton: the
    # factorisation is carried across Newton iterations and across load steps,
    # and only rebuilt when the residual stops dropping fast enough. A fresh
    # factorisation is forced at convergence of any plateau step, because the
    # adjoint must use the exact converged tangent.
    def newton(self, U, uin_val, Eabs, lu=None, need_lu=True):
        p, free = self.p, self.free
        U = U.copy()
        U[self.indofs] = uin_val
        refactor = lu is None
        nR_prev = np.inf
        for _ in range(p.max_nr):
            _, fhat, khat = self.element(U[self.edof], want_K=refactor)
            Fint = np.bincount(self.edof.ravel(),
                               weights=(fhat * Eabs[:, None]).ravel(),
                               minlength=self.ndof)
            Fint[self.outdofs] += self.ks * U[self.outdofs]
            R = Fint[free]
            nR = np.linalg.norm(R)
            if refactor:
                lu = splu(self.tangent(khat, Eabs).tocsc())
            if nR < p.tol_nr * max(1.0, np.linalg.norm(Fint)):
                if need_lu and not refactor:      # exact tangent for the adjoint
                    _, fhat, khat = self.element(U[self.edof], want_K=True)
                    lu = splu(self.tangent(khat, Eabs).tocsc())
                return U, lu, fhat, True
            dU = -lu.solve(R)
            alpha = 1.0                            # step-halving line search
            for _ls in range(6):
                Ut = U.copy()
                Ut[free] += alpha * dU
                F2, _ = self.internal_force(Ut, Eabs)
                if np.linalg.norm(F2[free]) < nR or alpha < 0.05:
                    break
                alpha *= 0.5
            U[free] += alpha * dU
            refactor = (nR > 0.35 * nR_prev)       # poor reduction -> rebuild
            nR_prev = nR
        return U, lu, fhat, False

    def step(self, U, uin_val, Eabs, lu=None, need_lu=True):
        """Newton, retried with a fresh factorisation then with sub-stepping."""
        Ut, lu2, fhat, ok = self.newton(U, uin_val, Eabs, lu, need_lu)
        if ok:
            return Ut, lu2, fhat, True
        if lu is not None:                          # stale factorisation? retry
            Ut, lu2, fhat, ok = self.newton(U, uin_val, Eabs, None, need_lu)
            if ok:
                return Ut, lu2, fhat, True
        u0 = float(U[self.indofs][0])
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
        """Walk the load path. Returns Fout (nstep,), dFout (nele, nstep), ok.

        States from the previous call are kept and reused as initial guesses.
        Between optimisation iterations the design barely moves, so this cuts
        Newton to two or three iterations per step after the first pass.
        """
        p = self.p
        if getattr(self, "_Upath", None) is None or len(self._Upath) != p.nstep:
            self._Upath = [np.zeros(self.ndof) for _ in range(p.nstep)]
        Fout = np.zeros(p.nstep)
        dFout = np.zeros((self.nele, p.nstep)) if dEabs is not None else None
        rhs = np.zeros(self.nfree)
        rhs[self.out_fr] = self.ks
        U, lu = np.zeros(self.ndof), None
        for k in range(1, p.nstep + 1):
            target = p.uin * k / p.nstep
            need_lu = dEabs is not None and k > p.nrise
            guess = self._Upath[k - 1]              # warm start, previous call
            Uk, lu2, fhat, ok = self.step(guess, target, Eabs, lu, need_lu)
            if not ok:                              # fall back to continuation
                Uk, lu2, fhat, ok = self.step(U, target, Eabs, None, need_lu)
                if not ok:
                    return Fout, dFout, False
            U, lu = Uk, lu2
            self._Upath[k - 1] = U.copy()
            Fout[k - 1] = self.ks * U[self.outdofs].sum()
            if need_lu:
                lam = np.zeros(self.ndof)
                lam[self.free] = lu.solve(rhs)      # reuse the LU factorisation
                dFout[:, k - 1] = -dEabs * np.einsum("ea,ea->e",
                                                     lam[self.edof], fhat)
        return Fout, dFout, True


# ----------------------------------------------------------------------------
def project(xTilde, beta, eta):
    tb = np.tanh(beta * eta)
    den = tb + np.tanh(beta * (1 - eta))
    xPhys = (tb + np.tanh(beta * (xTilde - eta))) / den
    dx = beta * (1 - np.tanh(beta * (xTilde - eta)) ** 2) / den
    return xPhys, dx


def optimise(p: Params | None = None, callback=None):
    p = p or Params()
    m = Model(p)
    x = np.full(m.nele, p.volfrac)
    beta, Ftar = p.beta, p.Ftar
    hist = {"J": [], "vol": [], "ripple": [], "Fout": []}

    if p.verbose:
        print(f"mesh {p.nelx}x{p.nely}x{p.nelz} = {m.nele} elements, "
              f"{m.ndof} dof ({m.nfree} free)")
        print(f"{'it':>4} {'J':>11} {'vol':>7} {'Fmean':>9} "
              f"{'ripple%':>8} {'change':>8} {'s/it':>7}")

    for it in range(1, p.maxiter + 1):
        t0 = time.time()
        xTilde = m.H @ x / m.Hs
        xPhys, dxdxt = project(xTilde, beta, p.eta)
        Eabs = p.E0 * (p.Emin + xPhys ** p.penal * (1 - p.Emin))
        dEabs = p.E0 * p.penal * xPhys ** (p.penal - 1) * (1 - p.Emin)

        Fout, dFout, ok = m.force_path(Eabs, dEabs)
        if not ok:
            print(f"  Newton failed at iteration {it}; reduce move or uin.")
            break

        P = slice(p.nrise, p.nstep)
        if Ftar is None:
            Ftar = float(np.median(Fout[P]))
            if abs(Ftar) < 1e-3 or np.any(np.sign(Fout[P]) != np.sign(Ftar)):
                raise RuntimeError(
                    f"initial plateau window is degenerate (Fout={Fout}). The "
                    "ports are too weakly coupled or the stroke is too small; "
                    "fix the load case before optimising.")
            if p.verbose:
                print(f"  auto target Ftar = {Ftar:.3f} N "
                      f"(median plateau of the uniform start)")

        r = (Fout[P] - Ftar) / Ftar
        nP = r.size
        J = float(r @ r / nP)
        dJ = dFout[:, P] @ (2 * r / (Ftar * nP))          # d/d xPhys

        dJ = m.H @ ((dJ * dxdxt) / m.Hs)                  # chain rule: filter
        dV = m.H @ (dxdxt / m.Hs) / m.nele

        # -- design update: projected gradient + bisection on the volume mult.
        dJn = dJ / max(np.abs(dJ).max(), 1e-30)
        dVn = dV / max(np.abs(dV).max(), 1e-30)
        lo, hi = -1e4, 1e4
        xnew = x
        while hi - lo > 1e-6:
            mid = 0.5 * (lo + hi)
            xnew = np.clip(np.clip(x - p.move * (dJn + mid * dVn),
                                   x - p.move, x + p.move), 0.0, 1.0)
            xp, _ = project(m.H @ xnew / m.Hs, beta, p.eta)
            if xp.mean() > p.volfrac:
                lo = mid
            else:
                hi = mid
        change = float(np.abs(xnew - x).max())
        x = xnew

        Fp = Fout[P]
        ripple = 100 * (Fp.max() - Fp.min()) / abs(Fp.mean())
        hist["J"].append(J); hist["vol"].append(float(xPhys.mean()))
        hist["ripple"].append(float(ripple)); hist["Fout"].append(Fout.copy())
        if p.verbose:
            print(f"{it:4d} {J:11.4e} {xPhys.mean():7.3f} {Fp.mean():9.3f} "
                  f"{ripple:8.2f} {change:8.4f} {time.time()-t0:7.1f}")
        if callback:
            callback(it, x, xPhys, Fout)

        if it % p.beta_iter == 0 and beta < p.beta_max:
            beta *= 2
            change = 1.0
            if p.verbose:
                print(f"  beta -> {beta:g}")
        if change < 0.002 and beta >= p.beta_max:
            break

    xTilde = m.H @ x / m.Hs
    xPhys, _ = project(xTilde, beta, p.eta)
    return xPhys, hist, m


def export_vtk(xPhys, m, path="design.vti", threshold=0.5):
    """Write a VTK ImageData file of the density field (open in ParaView)."""
    p = m.p
    rho = xPhys.reshape(p.nelz, p.nelx, p.nely).transpose(1, 2, 0)  # x,y,z
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="ImageData" '
                'version="0.1" byte_order="LittleEndian">\n')
        f.write(f'<ImageData WholeExtent="0 {p.nelx} 0 {p.nely} 0 {p.nelz}" '
                f'Origin="0 0 0" Spacing="{m.h} {m.h} {m.h}">\n')
        f.write(f'<Piece Extent="0 {p.nelx} 0 {p.nely} 0 {p.nelz}">\n')
        f.write('<CellData Scalars="density">\n'
                '<DataArray type="Float32" Name="density" format="ascii">\n')
        f.write(" ".join(f"{v:.4f}" for v in rho.transpose(2, 1, 0).ravel()))
        f.write('\n</DataArray>\n</CellData>\n</Piece>\n</ImageData>\n</VTKFile>\n')
    print(f"wrote {path}  (threshold at {threshold} in ParaView to see the shape)")


# ----------------------------------------------------------------------------
def verify():
    """Finite-difference checks of the element and the adjoint sensitivities."""
    rng = np.random.default_rng(0)
    p = Params(nelx=3, nely=2, nelz=2, Lx=0.006, nstep=5, nrise=1,
               uin=-4e-4, kout=1e3, tol_nr=1e-13, verbose=False)
    m = Model(p)
    x = rng.uniform(0.3, 0.9, m.nele)
    Eabs = p.E0 * (p.Emin + x ** p.penal * (1 - p.Emin))
    dEabs = p.E0 * p.penal * x ** (p.penal - 1) * (1 - p.Emin)

    U = np.zeros(m.ndof)
    U[m.free] = rng.normal(0, 3e-5, m.nfree)
    U[m.indofs] = -2e-4

    def total_energy(Uv):
        W, _, _ = m.element(Uv[m.edof], want_K=False, want_W=True)
        return float(W @ Eabs) + 0.5 * m.ks * float(Uv[m.outdofs] @ Uv[m.outdofs])

    # 1. fint == dW/du
    Fint, _ = m.internal_force(U, Eabs)
    d, num = 1e-9, np.zeros(m.nfree)
    for i, dof in enumerate(m.free):
        Up, Um = U.copy(), U.copy()
        Up[dof] += d; Um[dof] -= d
        num[i] = (total_energy(Up) - total_energy(Um)) / (2 * d)
    e1 = np.abs(num - Fint[m.free]).max() / np.abs(Fint[m.free]).max()

    # 2. Kt == dfint/du
    _, _, khat = m.element(U[m.edof])
    Kff = m.tangent(khat, Eabs).toarray()
    num = np.zeros((m.nfree, m.nfree))
    for i, dof in enumerate(m.free):
        Up, Um = U.copy(), U.copy()
        Up[dof] += d; Um[dof] -= d
        num[:, i] = (m.internal_force(Up, Eabs)[0][m.free]
                     - m.internal_force(Um, Eabs)[0][m.free]) / (2 * d)
    e2 = np.abs(num - Kff).max() / np.abs(Kff).max()

    # 3. adjoint dFout/dx
    m._Upath = None
    Fout, dFout, ok = m.force_path(Eabs, dEabs)
    assert ok, "load path failed during verification"
    worst = 0.0
    for e in rng.choice(m.nele, 4, replace=False):
        dd = 1e-7
        xp, xm = x.copy(), x.copy()
        xp[e] += dd; xm[e] -= dd
        Ep = p.E0 * (p.Emin + xp ** p.penal * (1 - p.Emin))
        Em = p.E0 * (p.Emin + xm ** p.penal * (1 - p.Emin))
        m._Upath = None; Fp_ = m.force_path(Ep)[0]
        m._Upath = None; Fm_ = m.force_path(Em)[0]
        fd = (Fp_ - Fm_) / (2 * dd)
        k0 = p.nrise
        worst = max(worst, np.abs(fd[k0:] - dFout[e, k0:]).max()
                    / np.abs(fd[k0:]).max())

    print(f"1. fint vs dW/du       rel err = {e1:.3e}")
    print(f"2. Kt   vs dfint/du    rel err = {e2:.3e}")
    print(f"3. adjoint dFout/dx    rel err = {worst:.3e}")
    print(f"   Fout path (N): {np.array2string(Fout, precision=3)}")
    return max(e1, e2, worst)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--time", type=int, default=0, metavar="N",
                    help="time N optimisation iterations")
    ap.add_argument("--vtk", type=str, default="", metavar="FILE")
    args = ap.parse_args()

    if args.verify:
        verify()
    if args.time:
        xPhys, hist, m = optimise(Params(maxiter=args.time))
    if args.run:
        xPhys, hist, m = optimise()
        if args.vtk:
            export_vtk(xPhys, m, args.vtk)
    if not (args.verify or args.run or args.time):
        ap.print_help()
