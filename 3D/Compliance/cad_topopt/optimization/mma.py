"""
Stage 5: Svanberg's globally convergent Method of Moving Asymptotes (GCMMA,
Svanberg 2002/2007) as the design update for the constant-force
optimisation, in place of top3d_cf.optimise()'s projected-gradient step.

Why not the projected-gradient step: it scales dJ to unit max and moves the
most sensitive elements by the full move limit on every iteration, whatever
the gradient says about how far away the optimum is. Near an optimum it
overshoots, reverses and overshoots again. On Part1 (100 iterations) J
flipped between ~0.07 and ~0.18 on alternate iterations and the density
never left its uniform start. top3d_cf.py's README calls that optimiser
"the weak link" and recommends MMA; Optimality Criteria is ruled out because
the sensitivities change sign across the plateau window.

Why not plain MMA either (tried first, on Part1 at a 5 mm stroke, with the
volume then only bounded from above -- see `optimise`): the volume bound was
not active -- a stiffer part overshoots Ftar, so the optimum did not use all
the material -- and with no active constraint MMA's step for each element depends only on its asymptotes, not
on how large its gradient is. Every element moved by the same amount, the
whole part got ~20% stiffer then ~20% softer on alternate iterations, and
once the asymptotes reached their minimum spacing (1% of the range) that
swing never died out: the output force kept alternating between 0.070 and
0.091 N around a 0.082 N target for 30+ iterations.

What GCMMA adds: after solving the MMA subproblem it evaluates the true
objective at the proposed design. If the convex approximation promised less
than the truth delivered (it was not "conservative"), it adds curvature to
the approximation (raa0 / raa, Svanberg's rho) and solves again from the same
point. Extra curvature makes each element's step proportional to its own
gradient, so the overshoot is removed at its cause, at the price of an extra
force-path solve on the iterations that need one. One departure from
Svanberg: the curvature is carried from one iteration into the next (see
`step`), which cut those extra solves from 4-8 to ~0.4 per iteration on a
test problem shaped like this one.

`optimise()` below is top3d_cf.optimise() line for line except for the blocks
marked CHANGED -- the physics (force_path and its adjoint, the filter, the
projection, the Ftar auto-target, the beta schedule, the printout) is the
same code, reorganised into one `evaluate(x)` so the inner loop can call it
at trial designs. It constructs `top3d_cf.Model(p)`, so inside
voxel_model.patched_model() it runs on the voxelized part, and outside it on
top3d_cf.py's own box problem. top3d_cf.py is still never modified.

`GCMMA` is a NumPy port of Svanberg's gcmmasub.m / asymp.m / raaupdate.m /
concheck.m with subsolv.m (the 2007 versions, m < n branch; here m = 2, the
volume held at volfrac from both sides), with those files' parameter values. `self_test()` checks
it against SciPy's SLSQP on two small problems with known unique optima;
optimize_main.py --verify runs it.
"""
from __future__ import annotations

import time

import numpy as np

from .voxel_model import top3d_cf

# Svanberg's defaults (gcmmasub.m, asymp.m, raaupdate.m).
EPSIMIN = 1e-7      # subproblem barrier floor, and the conservativeness slack
RAAEPS = 1e-6       # smallest curvature term
ALBEFA = 0.1        # a step may cover at most 90% of the gap to an asymptote
ASYINIT = 0.5       # first two iterations: asymptotes at x -/+ 0.5 * range
ASYINCR = 1.2       # widen after two moves in the same direction
ASYDECR = 0.7       # tighten after a reversal
MAX_INNER = 15      # proposals per iteration before the last one is taken anyway
# How far the truth may exceed the approximation and still count as
# conservative. Svanberg uses EPSIMIN (1e-7), which assumes f is evaluated
# exactly. Here f0 = J/J_start comes out of a Newton solve converged to a
# relative residual of 1e-7, which leaves roughly 1e-5 of noise in it; any
# tighter and GCMMA rejects steps on that noise, each rejection costing one
# more force-path solve.
CONS_TOL = 1e-5


class GCMMA:
    """Svanberg's GCMMA for

        min f0(x)   s.t.  f_i(x) <= 0 (i = 1..m),   xmin <= x <= xmax

    posed with his standard NLP settings a0 = 1, a_i = 0, c_i = 1000,
    d_i = 1: the artificial variables y_i only become non-zero if a
    subproblem cannot satisfy the linearised constraints at all.

    Holds what carries over between iterations (the last two iterates and
    the asymptotes). Call `step` once per design iteration.
    """

    def __init__(self, n: int, m: int, xmin, xmax, move: float):
        if m >= n:
            raise ValueError("only the m < n form of the MMA subproblem is implemented")
        self.n, self.m = n, m
        self.xmin = np.broadcast_to(np.asarray(xmin, float), (n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(xmax, float), (n,)).copy()
        self.span = np.maximum(self.xmax - self.xmin, 1e-5)
        self.move = move
        self.a0 = 1.0
        self.a = np.zeros(m)
        self.c = np.full(m, 1000.0)
        self.d = np.ones(m)
        self.k = 0
        self.xold1 = self.xold2 = self.low = self.upp = None
        self.raa0_last = self.raa_last = None

    def step(self, x, f0val, df0dx, fval, dfdx, evaluate):
        """One outer iteration from x.

        f0val scalar, df0dx (n,), fval (m,), dfdx (m, n): the problem at x.
        evaluate(xtrial) -> (f0, fval, payload), or None if the problem
        cannot be evaluated there (e.g. Newton failed). payload is handed back
        untouched for the accepted design, so the caller can reuse whatever
        it computed there (here: the gradients for the next iteration).

        Returns (xnew, payload, n_evaluations). payload is None only if
        every trial failed to evaluate; xnew is then x.
        """
        x = np.asarray(x, float)
        fval = np.asarray(fval, float)
        dfdx = np.asarray(dfdx, float).reshape(self.m, self.n)
        self._asymptotes(x)

        # initial curvature: a tenth of the mean gradient magnitude (asymp.m)
        raa0 = max(RAAEPS, 0.1 / self.n * (np.abs(df0dx) @ self.span))
        raa = np.maximum(RAAEPS, 0.1 / self.n * (np.abs(dfdx) @ self.span))
        # ... or half of what the previous iteration ended on, if larger.
        # Svanberg restarts from the gradient every iteration, so on a
        # problem this sharply curved every iteration re-learns its caution
        # through 4-8 rejected proposals (one force-path solve each). Starting
        # from the last value, halved so it can relax, got a test problem to
        # the same optimum at the same pace with 1.4 solves per iteration
        # instead of 5-8. Acceptance is still checked, so it can only make
        # the first proposal more cautious, never let a bad one through.
        if self.raa0_last is not None:
            raa0 = max(raa0, 0.5 * self.raa0_last)
            raa = np.maximum(raa, 0.5 * self.raa_last)

        xnew, payload = x, None
        for inner in range(1, MAX_INNER + 1):
            xnew, f0app, fapp = self._subproblem(x, f0val, df0dx, fval, dfdx, raa0, raa)
            res = evaluate(xnew)
            if res is None:                       # could not even evaluate: be
                raa0, raa = 10 * raa0, 10 * raa   # far more careful, try again
                continue
            f0new, fnew, payload = res
            fnew = np.asarray(fnew, float)
            if f0app + CONS_TOL >= f0new and np.all(fapp + CONS_TOL >= fnew):
                break                             # conservative: accept
            raa0, raa = self._raise_curvature(x, xnew, f0new, fnew, f0app, fapp,
                                              raa0, raa)
        if payload is None:
            return x, None, inner

        self.raa0_last, self.raa_last = raa0, raa
        self.xold2 = self.xold1
        self.xold1 = x.copy()
        return xnew, payload, inner

    # ------------------------------------------------------------------ parts
    def _asymptotes(self, x):
        """asymp.m: the same asymptote rules as plain MMA."""
        self.k += 1
        span = self.span
        if self.k <= 2:
            self.low = x - ASYINIT * span
            self.upp = x + ASYINIT * span
            return
        trend = (x - self.xold1) * (self.xold1 - self.xold2)
        factor = np.ones(self.n)
        factor[trend > 0] = ASYINCR
        factor[trend < 0] = ASYDECR
        low = x - factor * (self.xold1 - self.low)
        upp = x + factor * (self.upp - self.xold1)
        self.low = np.clip(low, x - 10 * span, x - 0.01 * span)
        self.upp = np.clip(upp, x + 0.01 * span, x + 10 * span)

    def _subproblem(self, x, f0val, df0dx, fval, dfdx, raa0, raa):
        """gcmmasub.m: build the convex approximations with curvature
        raa0/raa, solve, and return the new x with the approximations'
        predicted values there (what `step` checks against the truth)."""
        low, upp, span = self.low, self.upp, self.span
        alfa = np.maximum.reduce([low + ALBEFA * (x - low), x - self.move * span, self.xmin])
        beta = np.minimum.reduce([upp - ALBEFA * (upp - x), x + self.move * span, self.xmax])

        ux1, xl1 = upp - x, x - low
        ux2, xl2 = ux1 * ux1, xl1 * xl1
        uxinv, xlinv = 1.0 / ux1, 1.0 / xl1

        # objective: p/(U - x) + q/(x - L), gradient split by sign
        p0 = np.maximum(df0dx, 0.0)
        q0 = np.maximum(-df0dx, 0.0)
        pq0 = p0 + q0
        p0 = (p0 + 0.001 * pq0 + raa0 / span) * ux2
        q0 = (q0 + 0.001 * pq0 + raa0 / span) * xl2
        r0 = f0val - p0 @ uxinv - q0 @ xlinv

        # constraints, the same way
        P = np.maximum(dfdx, 0.0)
        Q = np.maximum(-dfdx, 0.0)
        PQ = P + Q
        P = (P + 0.001 * PQ + raa[:, None] / span[None, :]) * ux2[None, :]
        Q = (Q + 0.001 * PQ + raa[:, None] / span[None, :]) * xl2[None, :]
        r = fval - P @ uxinv - Q @ xlinv

        xnew, _ = _subsolv(self.m, low, upp, alfa, beta, p0, q0, P, Q,
                           self.a0, self.a, -r, self.c, self.d)
        uxn, xln = 1.0 / (upp - xnew), 1.0 / (xnew - low)
        return xnew, r0 + p0 @ uxn + q0 @ xln, r + P @ uxn + Q @ xln

    def _raise_curvature(self, x, xnew, f0new, fnew, f0app, fapp, raa0, raa):
        """raaupdate.m: grow the curvature of every function whose
        approximation was not conservative, by just enough (x1.1) that it
        would have been, capped at 10x per inner iteration."""
        low, upp = self.low, self.upp
        cof = ((xnew - x) / (upp - xnew)) * ((xnew - x) / (xnew - low)) @ ((upp - low) / self.span)
        cof = max(cof, 1e-12)
        if f0new > f0app + 0.5 * EPSIMIN:
            raa0 = min(1.1 * (raa0 + (f0new - f0app) / cof), 10 * raa0)
        grow = fnew > fapp + 0.5 * EPSIMIN
        raa = np.where(grow, np.minimum(1.1 * (raa + (fnew - fapp) / cof), 10 * raa), raa)
        return raa0, raa


def _subsolv(m, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    """Svanberg's subsolv.m: primal-dual interior point on the MMA
    subproblem, with the barrier parameter lowered 10x per outer pass.
    Only the m < n branch (an (m+1) x (m+1) reduced system)."""
    x = 0.5 * (alfa + beta)
    y = np.ones(m)
    z = 1.0
    lam = np.ones(m)
    xsi = np.maximum(1.0 / (x - alfa), 1.0)
    eta = np.maximum(1.0 / (beta - x), 1.0)
    mu = np.maximum(1.0, 0.5 * c)
    zet = 1.0
    s = np.ones(m)

    def residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi):
        ux1, xl1 = upp - x, x - low
        plam = p0 + P.T @ lam
        qlam = q0 + Q.T @ lam
        gvec = P @ (1.0 / ux1) + Q @ (1.0 / xl1)
        return np.concatenate([
            plam / ux1 ** 2 - qlam / xl1 ** 2 - xsi + eta,      # rex
            c + d * y - mu - lam,                                # rey
            [a0 - zet - a @ lam],                                # rez
            gvec - a * z - y + s - b,                            # relam
            xsi * (x - alfa) - epsi,                             # rexsi
            eta * (beta - x) - epsi,                             # reeta
            mu * y - epsi,                                       # remu
            [zet * z - epsi],                                    # rezet
            lam * s - epsi])                                     # res

    epsi = 1.0
    while epsi > EPSIMIN:
        r = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
        residunorm, residumax = np.linalg.norm(r), np.abs(r).max()
        ittt = 0
        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1
            ux1, xl1 = upp - x, x - low
            ux2, xl2 = ux1 * ux1, xl1 * xl1
            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ (1.0 / ux1) + Q @ (1.0 / xl1)
            GG = P / ux2[None, :] - Q / xl2[None, :]
            delx = plam / ux2 - qlam / xl2 - epsi / (x - alfa) + epsi / (beta - x)
            dely = c + d * y - lam - epsi / y
            delz = a0 - a @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsi / lam
            diagx = (2.0 * (plam / (ux2 * ux1) + qlam / (xl2 * xl1))
                     + xsi / (x - alfa) + eta / (beta - x))
            diagy = d + mu / y
            diaglamyi = s / lam + 1.0 / diagy

            blam = dellam + dely / diagy - GG @ (delx / diagx)
            AA = np.zeros((m + 1, m + 1))
            AA[:m, :m] = np.diag(diaglamyi) + (GG / diagx[None, :]) @ GG.T
            AA[:m, m] = a
            AA[m, :m] = a
            AA[m, m] = -zet / z
            sol = np.linalg.solve(AA, np.concatenate([blam, [delz]]))
            dlam, dz = sol[:m], sol[m]
            dx = -delx / diagx - (GG.T @ dlam) / diagx
            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsi / (x - alfa) - xsi * dx / (x - alfa)
            deta = -eta + epsi / (beta - x) + eta * dx / (beta - x)
            dmu = -mu + epsi / y - mu * dy / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsi / lam - s * dlam / lam

            # largest step keeping every slack/multiplier and x-alfa, beta-x
            # positive (with a 1% margin), capped at a full Newton step
            xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
            dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])
            stminv = max((-1.01 * dxx / xx).max(),
                         (-1.01 * dx / (x - alfa)).max(),
                         (1.01 * dx / (beta - x)).max(), 1.0)
            steg = 1.0 / stminv

            old = (x, y, z, lam, xsi, eta, mu, zet, s)
            step = (dx, dy, dz, dlam, dxsi, deta, dmu, dzet, ds)
            itto, resinew = 0, 2 * residunorm
            while resinew > residunorm and itto < 50:    # step halving
                itto += 1
                x, y, z, lam, xsi, eta, mu, zet, s = (
                    v + steg * dv for v, dv in zip(old, step))
                r = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
                resinew = np.linalg.norm(r)
                steg /= 2
            residunorm, residumax = resinew, np.abs(r).max()
        epsi *= 0.1
    return x, y


# ----------------------------------------------------------------------------
def optimise(p, callback=None):
    """top3d_cf.optimise() with the design update replaced by GCMMA."""
    m = top3d_cf.Model(p)
    x = np.full(m.nele, p.volfrac)
    beta, Ftar = p.beta, p.Ftar
    Jstart = None
    P = slice(p.nrise, p.nstep)
    hist = {"J": [], "vol": [], "ripple": [], "Fout": []}
    # CHANGED: GCMMA carries its asymptotes and last two iterates between
    # iterations. Two constraints: the volume from above and from below.
    opt = GCMMA(n=m.nele, m=2, xmin=0.0, xmax=1.0, move=p.move)

    # CHANGED: the body of top3d_cf.optimise()'s loop, from the filter down
    # to the chain rule, as one function -- GCMMA's inner loop has to run it
    # at trial designs, not only once per iteration.
    def evaluate(x):
        nonlocal Ftar, Jstart
        xTilde = m.H @ x / m.Hs
        xPhys, dxdxt = top3d_cf.project(xTilde, beta, p.eta)
        Eabs = p.E0 * (p.Emin + xPhys ** p.penal * (1 - p.Emin))
        dEabs = p.E0 * p.penal * xPhys ** (p.penal - 1) * (1 - p.Emin)

        Fout, dFout, ok = m.force_path(Eabs, dEabs)
        if not ok:
            return None

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
        # objective   J / J_start: starts at 1, so it is on the same O(1)
        #             scale as the constraint and small next to the penalty
        #             c = 1000 on an unmet one. (ln J was tried: it curves
        #             so sharply near a small minimum that GCMMA needed 2-3x
        #             more force-path solves per iteration.)
        # constraints mean(xPhys) = volfrac, as two inequalities (<= and
        #             >=), scaled to O(1). An equality, not just an upper
        #             bound, because that is what top3d_cf.optimise()'s
        #             bisection enforces: its multiplier runs over -1e4..1e4,
        #             so it adds material as readily as it removes it (its
        #             log shows vol = 0.300 on every row). With only "<=",
        #             this objective -- which does not reward material --
        #             let the volume drift to 0.27, the field stayed grey
        #             below the eta = 0.5 threshold, and each beta doubling
        #             projected more of it away: vol 0.24, 0.15, 0.04 at
        #             beta 2, 4, 8.
        if Jstart is None:
            Jstart = max(J, 1e-12)
        g = xPhys.mean() / p.volfrac - 1.0
        dg = dV / p.volfrac
        return dict(xPhys=xPhys, Fout=Fout, J=J, f0=J / Jstart, df0=dJ / Jstart,
                    g=np.array([g, -g]), dg=np.stack([dg, -dg]))

    def trial(xt):
        ev = evaluate(xt)
        return None if ev is None else (ev["f0"], ev["g"], ev)

    if p.verbose:
        print(f"mesh {p.nelx}x{p.nely}x{p.nelz} = {m.nele} elements, "
              f"{m.ndof} dof ({m.nfree} free)")
        print(f"optimiser: GCMMA, move limit {p.move}")    # CHANGED
        print(f"{'it':>4} {'J':>11} {'vol':>7} {'Fmean':>9} "
              f"{'ripple%':>8} {'change':>8} {'s/it':>7} {'solves':>6}")

    ev = evaluate(x)
    if ev is None:
        print("  Newton failed on the starting design; reduce uin.")
        return top3d_cf.project(m.H @ x / m.Hs, beta, p.eta)[0], hist, m

    for it in range(1, p.maxiter + 1):
        t0 = time.time()
        # CHANGED: one GCMMA outer iteration instead of the normalised
        # projected-gradient step and its bisection. It evaluates each
        # proposal and hands back the accepted one's evaluation, which is
        # the starting point of the next iteration.
        xnew, ev_new, solves = opt.step(x, ev["f0"], ev["df0"], ev["g"], ev["dg"], trial)
        change = float(np.abs(xnew - x).max())

        Fp = ev["Fout"][P]
        ripple = 100 * (Fp.max() - Fp.min()) / abs(Fp.mean())
        vol = float(ev["xPhys"].mean())
        hist["J"].append(ev["J"]); hist["vol"].append(vol)
        hist["ripple"].append(float(ripple)); hist["Fout"].append(ev["Fout"].copy())
        if p.verbose:
            print(f"{it:4d} {ev['J']:11.4e} {vol:7.3f} {Fp.mean():9.3f} "
                  f"{ripple:8.2f} {change:8.4f} {time.time()-t0:7.1f} {solves:6d}")
        if callback:
            callback(it, xnew, ev["xPhys"], ev["Fout"])
        if ev_new is None:
            print(f"  Newton failed at every trial design in iteration {it}; "
                  f"reduce move or uin.")
            break
        x, ev = xnew, ev_new

        if it % p.beta_iter == 0 and beta < p.beta_max:
            beta *= 2
            change = 1.0
            # CHANGED: a sharper projection is a different function of x, so
            # the accepted evaluation is stale; redo it. GCMMA's asymptote
            # history is kept -- restarting it (tried) brings back the large
            # early swings at every beta step.
            ev = evaluate(x)
            if ev is None:
                print(f"  Newton failed after beta -> {beta:g}; reduce move or uin.")
                break
            if p.verbose:
                print(f"  beta -> {beta:g}")
        if change < 0.002 and beta >= p.beta_max:
            break

    xTilde = m.H @ x / m.Hs
    xPhys, _ = top3d_cf.project(xTilde, beta, p.eta)
    return xPhys, hist, m


# ----------------------------------------------------------------------------
def self_test(verbose: bool = True) -> float:
    """GCMMA vs SciPy's SLSQP on two convex problems with unique optima.

    1. Svanberg's own 3-variable test problem (two quadratic constraints);
       his published optimum is x = (2.0175, 1.7800, 1.2375).
    2. The shape of the real problem, scaled down: 200 variables in [0, 1],
       a strictly convex objective whose gradient has both signs and couples
       the variables, an active mean-"volume" bound, and move limit 0.05 as
       in config.py. Part of the optimum sits on the 0/1 bounds, part inside.

    Returns the worst error of the GCMMA solution against SLSQP's (max |dx|,
    relative to max |x|). The floor is ~1e-6: the subproblem's barrier
    parameter stops at EPSIMIN = 1e-7.
    """
    from scipy.optimize import minimize

    def solve_both(f0, g, x0, lo, hi, move, iters):
        opt = GCMMA(n=x0.size, m=len(g(x0)[0]), xmin=lo, xmax=hi, move=move)
        x = x0.copy()
        for _ in range(iters):
            xn, _, _ = opt.step(x, f0(x)[0], f0(x)[1], *g(x),
                                lambda v: (f0(v)[0], g(v)[0], True))
            done = np.abs(xn - x).max() < 1e-8
            x = xn
            if done:
                break
        res = minimize(lambda v: f0(v)[0], x0, jac=lambda v: f0(v)[1],
                       method="SLSQP", bounds=list(zip(lo, hi)),
                       constraints={"type": "ineq", "fun": lambda v: -g(v)[0],
                                    "jac": lambda v: -g(v)[1]},
                       options={"ftol": 1e-12, "maxiter": 1000})
        assert res.success, f"SLSQP reference failed: {res.message}"
        return np.abs(x - res.x).max() / np.abs(res.x).max(), res.x

    # 1. Svanberg's test problem
    c1, c2 = np.array([5.0, 2.0, 1.0]), np.array([3.0, 4.0, 3.0])
    err1, _ = solve_both(
        lambda x: (x @ x, 2 * x),
        lambda x: (np.array([(x - c1) @ (x - c1) - 9, (x - c2) @ (x - c2) - 9]),
                   np.stack([2 * (x - c1), 2 * (x - c2)])),
        np.array([4.0, 3.0, 2.0]), np.zeros(3), np.full(3, 5.0), move=0.5, iters=200)

    # 2. topopt-shaped: sum c/(x+0.1) pulls material in (like compliance),
    #    a random-signed linear term pushes some of it out, B couples.
    rng = np.random.default_rng(0)
    n, vf = 200, 0.3
    c = rng.uniform(0.01, 1.0, n)
    s = rng.normal(0.0, 3.0, n)
    B = rng.normal(size=(n, n)) / n

    def f0(x):
        Bx = B @ x
        return ((c / (x + 0.1)).sum() + s @ x + 0.5 * Bx @ Bx,
                -c / (x + 0.1) ** 2 + s + B.T @ Bx)

    err2, xs = solve_both(
        f0, lambda x: (np.array([x.mean() / vf - 1]), np.full((1, n), 1 / (n * vf))),
        np.full(n, vf), np.zeros(n), np.ones(n), move=0.05, iters=300)

    if verbose:
        at_bound = int(((xs < 1e-6) | (xs > 1 - 1e-6)).sum())
        print(f"1. Svanberg test problem   x err vs SLSQP = {err1:.3e}")
        print(f"2. 200-var volume problem  x err vs SLSQP = {err2:.3e}"
              f"   ({at_bound} of {n} at a 0/1 bound)")
    return max(err1, err2)
