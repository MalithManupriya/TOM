# Constant-force topology optimisation — `top2d_cf.m`

## Why this is not a modified `top3d`

`top3d` solves minimum compliance: find the stiffest layout at a given volume. Three things
have to change for a constant-force mechanism, and none of them is optional.

| | `top3d` | `top2d_cf` |
|---|---|---|
| Objective | `c = UᵀKU`, minimise | `Σ (F_out − F*)² / F*²` over a plateau window |
| Analysis | one linear solve | displacement-controlled Newton–Raphson, one per load step |
| Kinematics | small strain, `F = Ku` | total Lagrangian, neo-Hookean, geometrically nonlinear |
| Update | Optimality Criteria | MMA (OC assumes a monotone objective; this one isn't) |

The kinematics point is the one worth internalising. Under linear assumptions the output force
is proportional to displacement, so a plateau is not merely hard to find — it does not exist in
the model. The flat region is a large-deformation / post-buckling effect, which is exactly the
argument your review makes in §2.3.1.

## What the code solves

```
min_x   J = (1/|P|) Σ_{k∈P} [ (F_out,k − F*) / F* ]²
s.t.    V(x)/V₀ ≤ volfrac
        R(U_k, x) = 0     for every load step k
        0 ≤ x ≤ 1
```

`P` is the plateau window, load steps `nRise+1 … nStep`. Steps `1 … nRise` are the rise region
and are deliberately left free — every physical constant-force mechanism needs a preload stroke
before the plateau (Xia et al.: 2.2 mm preload, then 6.4 mm flat). Constraining the rise region
would ask for a step discontinuity at `u = 0` and the optimiser will chase it forever.

## Model

```
     |<------------------ Lx ------------------>|
     +------------------------------------------o   output dof (vertical),
  // |                                          |   grounded through spring k_out
  // |              design domain               |   → this spring IS the tissue
  // |                                          |
  // |                                          o   input dof (horizontal),
  // +------------------------------------------+   prescribed stroke u_in
   clamped
```

Output force is read as `F_out = k_out · u_out`. Setting `k_out` from your simulant
(E ≈ 100–300 kPa) is what couples the optimisation to the clinical parameters in §2.1.2.5
rather than leaving it as an abstract force target.

## Running it

```matlab
top2d_cf                                  % defaults, ~30×15 mm domain
p.nelx = 90; p.nely = 45; p.Ftar = 2.0;   % override anything
[xPhys, hist] = top2d_cf(p);
```

Leave `p.Ftar = []` on the first run. The code takes the median plateau force of the uniform
starting design and prints it. Check that number is in the range you want before committing to
a long run — if the uniform design gives 40 N and you asked for 2 N, the optimiser has to
remove most of the structure to get there and will probably produce something spindly. Adjust
`E0`, `thk` or `uin` until the starting design is within roughly 2–3× of your target.

Key parameters:

| Parameter | Meaning | Tuning note |
|---|---|---|
| `uin` | total input stroke (negative = pushing in) | too small → no buckling → no plateau |
| `nStep` / `nRise` | load steps / rise region | more steps = better plateau resolution, linear cost |
| `kout` | output spring (N/m) | your tissue simulant contact stiffness |
| `Ftar` | plateau force (N) | 1.0–3.0 N for liver, up to 5.34 N for GI |
| `rmin` | filter radius, elements | ≥ 2.5 keeps printable feature sizes |
| `move` | move limit | drop to 0.02 if the force path oscillates |

Reported each iteration: objective, volume, mean plateau force, and **ripple %** —
`(F_max − F_min)/F_mean` over the plateau. That is directly comparable to the 0.9 % / 0.7 % / ≤2 %
figures in your Table 8.

## Optimiser

The code uses Svanberg's `mmasub.m` if it is on the path, which is what you want for final
results. MMA is free for academic use on request from the author; nearly every paper in your
Table 8 uses it. Check the argument list of whichever version you obtain — some releases take
extra second-derivative arguments.

Without it, the code falls back to a projected-gradient step with bisection on the volume
multiplier. It converges and it is honest, but it is slower and more sensitive to `move`.

## Verification

Element internal force, tangent stiffness and adjoint sensitivities were checked against finite
differences on a small mesh:

```
fint vs dW/du        rel err 2.1e-09
Kt   vs dfint/du     rel err 4.9e-10
adjoint dFout/dx     rel err 1.5e-07
```

If you modify the element routine, re-run that check before trusting any optimisation result.
Wrong sensitivities do not crash — they quietly converge to the wrong shape.

## Known limitations (state these in your report)

1. **Void element instability.** Low-density elements can invert under large deformation and
   pollute the tangent. The code clamps `det C` as a crude guard. The proper fix is the energy
   interpolation scheme of Wang, Lazarov, Sigmund & Jensen (2014), which blends linear and
   nonlinear strain measures element-by-element. Worth adding before you publish numbers.
2. **No stress constraint.** Topology-optimised flexures develop thin, high-strain members.
   Your review already flags this. Add a p-norm stress constraint, or at minimum post-check
   peak stress in nonlinear FEA and reject designs that exceed the printed material's limit.
3. **No bifurcation constraint.** Chen et al.'s point stands: a flat force objective alone can
   produce a structure that is mechanically unstable. Add a lowest-eigenvalue constraint on the
   tangent stiffness if you see snap-through in the converged design.
4. **No contact model.** The tissue is a linear grounded spring, not a deformable body. Reddy
   et al. (your ref [3] in §2.3) argue contact interaction changes the force transfer materially.
   Adding it is a significant extension; the spring is a defensible first approximation.
5. **Single load case.** Tissue thickness varies between grasps. Consider optimising over two
   or three `kout` values simultaneously for robustness.

## Route to 3D

Your project is explicitly 3D monolithic, and your own gap statement criticises "simplified 2D
structural extrusions". So this 2D code is a stage, not the destination. Get the formulation
working here first — it is identical in 3D and far cheaper to debug.

What changes:

- **Element**: Q4 → H8. `dNdX` becomes 3×8, `B` becomes 6×24, `D` becomes 6×6, the geometric
  stiffness block becomes 3×3 identity-scaled. The neo-Hookean stress and tangent expressions
  are unchanged in form — only the index ranges grow.
- **Filter**: add the `k` loop, as in `top3d`. Note that `top3d` has a preallocation bug here
  (it uses `^2` where 3D needs `^3`); fix it or the arrays grow dynamically.
- **Solver**: this is where it hurts. You now have a Newton–Raphson loop (say 5 iterations)
  inside every load step (12), inside every optimisation iteration (150) — roughly 9000 3D
  sparse solves. Direct factorisation will not scale. Move to PCG with a multigrid or
  incomplete-Cholesky preconditioner, and reuse the previous step's solution as the initial
  guess. Expect hours, not minutes.
- **Symmetry**: exploit it. A grasper jaw is usually symmetric about one plane, which halves
  the model for free.

Budget realistically. A coarse 3D mesh (say 40×20×10) that runs overnight is far more useful
than a fine one that never converges before your deadline.
