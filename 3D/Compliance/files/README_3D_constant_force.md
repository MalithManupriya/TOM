# 3-D constant-force topology optimisation

Two implementations of the same formulation:

- `top3d_cf.py` — Python/NumPy/SciPy. **Tested.** Element routine and adjoint
  sensitivities verified against finite differences; optimisation run and timed.
- `top3d_cf.m` — MATLAB. A line-for-line translation of the Python. **Not
  executed** (no MATLAB available where this was written), so treat the Python
  as the reference and expect to shake out syntax on first run.

Both need R2020b+ (MATLAB, for `pagemtimes`) or NumPy + SciPy (Python).

## Verification

```
$ python3 top3d_cf.py --verify
1. fint vs dW/du       rel err = 7.666e-10     internal force is the exact gradient of strain energy
2. Kt   vs dfint/du    rel err = 9.469e-11     tangent is the exact Jacobian
3. adjoint dFout/dx    rel err = 4.765e-08     sensitivities match finite differences
```

Re-run this after **any** edit to the element routine. Wrong sensitivities do
not crash. They converge quietly to the wrong shape, and you will not find out
until you print the part.

## What changed from 2-D

Mechanically, not much: Q4 → H8, 8 DOF → 24, 3-component Voigt → 6, 4 Gauss
points → 8. The neo-Hookean stress and tangent expressions are unchanged in
form. What changes is cost, and two things had to be fixed before the code was
usable at all.

### 1. The load case has to couple the ports

The obvious extension of the 2-D layout — push in x at the mid-right edge, read
force out in y at the top-right — does not work in 3-D. Measured initial force
path on a uniform design:

```
A  mid-right x -> top-right y    F = [-0.005 -0.011 -0.014 -0.011  0.031  1.472 -0.200 -0.183]
B  bot-right x -> top-right x    F = [ 0.073  0.144  0.212  0.275  0.332  0.384  0.429  0.468]
C  left-mid  x -> right-mid  x   F = [ 0.117  0.236  0.359  0.485  0.615  0.749  0.886  1.027]
D  right face x -> top-right y   F = [ 0.017  0.034  0.051  0.067  0.082  0.097  0.108  0.116]
```

Case A is non-monotone and sign-indefinite, the auto target comes out negative,
and the optimiser diverges within three iterations. Case B is the shipped
default: monotone, and already *softening* on a uniform block (increments 0.071,
0.068, 0.063, 0.057, 0.052, 0.045, 0.039 N). That softening is the plateau
trying to form on its own, which is exactly the starting point you want. Case C
couples even harder but *stiffens*, so the optimiser has to fight the natural
response.

The general lesson: put both ports on the free end acting along the same axis.
Cross-direction layouts rely on Poisson coupling and bending, which is far too
weak on a uniform start. The code now raises an error rather than optimising
against a degenerate initial path.

### 2. Factorisation dominates, so the solver is modified Newton

Measured on the default mesh (32×12×6 = 2304 elements, 9009 DOF):

```
element (with tangent)  0.114 s
tangent assembly        0.010 s
sparse LU (splu)        0.464 s      <- 75% of the cost
internal force only     0.037 s
```

So the code carries the factorisation across Newton iterations *and* across load
steps, rebuilding only when the residual stops dropping by 3× per iteration. A
fresh factorisation is forced at convergence of each plateau step, because the
adjoint must use the exact converged tangent. States are also cached and reused
as initial guesses on the next optimisation iteration — between iterations the
design barely moves.

Net effect, measured:

```
   it           J     vol     Fmean  ripple%   change    s/it
    1  2.0241e-01   0.286     0.338   141.75   0.0500   175.8    <- cold start
    2  1.7775e-01   0.300     0.321   132.61   0.0500    15.6
    3  1.6183e-01   0.300     0.318   124.39   0.0500    15.8
    ...
    8  8.4967e-02   0.300     0.341    86.65   0.0438    21.9
```

94 s/iteration before these changes, ~15 s after. Budget roughly 30 minutes for
100 iterations on the default mesh. Ripple falls steadily but needs a few
hundred iterations with the fallback optimiser to reach single digits — see
below.

## Running

```bash
python3 top3d_cf.py --verify          # finite-difference checks
python3 top3d_cf.py --time 5          # time 5 iterations
python3 top3d_cf.py --run --vtk design.vti   # full run, ParaView output
```

```matlab
top3d_cf
q.nelx = 40; q.nelz = 8; q.Ftar = 2.0;
[xPhys, hist] = top3d_cf(q);
```

Leave `Ftar` empty on the first run. The code takes the median plateau force of
the uniform start and prints it. **Check that number before committing to a long
run.** If the uniform design gives 0.4 N and you want 2 N, the optimiser has to
stiffen dramatically and will likely fail. Fix the scale first — increase `nelz`
(width), `E0`, or `volfrac`, or lengthen `uin`. Getting the start within roughly
2–3× of your target is the single highest-leverage thing you can do.

Reported per iteration: objective, volume, mean plateau force, and **ripple %**
= `(Fmax − Fmin)/|Fmean|` over the plateau window. That is directly comparable
to the 0.9% / 0.7% / ≤2% figures in your Table 8.

## The optimiser is the weak link

Both files ship a projected-gradient step with bisection on the volume
multiplier. It is correct and it runs out of the box, but it is a first-order
method with a fixed move limit, and it is why ripple is still 87% after eight
iterations.

Optimality Criteria is not an option here — it assumes single-signed
sensitivities, and these flip sign depending on whether a load step sits above
or below the target.

For anything you intend to report, use **MMA**. Svanberg's `mmasub.m` is free
for academic use on request from the author; in Python, `nlopt`'s `LD_MMA` takes
the same objective and gradient. Expect convergence in tens rather than hundreds
of iterations. Every paper in your Table 8 uses MMA or a comparable
sequential-convex method.

## Known limitations — state these in your report

1. **Void element instability.** Low-density elements can invert under large
   deformation. Both codes clamp `det C` as a crude guard. The proper fix is the
   energy interpolation scheme of Wang, Lazarov, Sigmund & Jensen (2014). This
   matters more in 3-D than 2-D, and is worth adding before publishing numbers.
2. **No stress constraint.** Topology-optimised flexures develop thin,
   high-strain members. At minimum, post-check peak stress in nonlinear FEA and
   reject designs above the printed material limit; better, add a p-norm stress
   constraint.
3. **No bifurcation constraint.** Chen et al.'s point stands: a flat force
   objective alone can produce a mechanically unstable structure. Add a
   lowest-eigenvalue constraint on the tangent if you see snap-through.
4. **Tissue is a linear grounded spring.** Not a deformable contacting body.
   Reddy et al. argue contact interaction changes force transfer materially.
   The spring is a defensible first approximation; say so explicitly.
5. **Single load case.** Tissue thickness varies between grasps. Consider
   optimising over two or three `kout` values simultaneously for robustness.
6. **No minimum-feature guarantee.** The density filter plus Heaviside
   projection gives approximate length-scale control, not a hard bound. Check
   the thinnest member against your printer's resolution before fabricating.

## Scaling up

The direct sparse solver is fine to roughly 50k DOF. Beyond that, switch to
preconditioned CG with the previous step's solution as the initial guess —
that is the standard route and it is what `top3d`'s own successors do. Also
exploit symmetry: a grasper jaw is usually symmetric about one plane, which
halves the model for free.

A coarse mesh that runs overnight and converges is worth more than a fine one
that misses your deadline.
