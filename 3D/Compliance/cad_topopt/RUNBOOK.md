# cad_topopt — runbook & status

Persistent state for this project. **Read this first in a new session instead of
re-reading every source file.** Update the "Last verified" table whenever you
re-run a stage; update "Open issues" when one is opened or closed.

- Project dir: `3D/Compliance/cad_topopt`
- Physics engine: `../files/top3d_cf.py` — **never edited**, only imported and
  subclassed (`optimization/voxel_model.py` monkeypatches `Model` for the
  duration of the call).
- All tunables live in `config.py`. No other file should need editing to run a
  different part.

## Environment

Verified working on Windows 11, system Python (no venv):

```
Python 3.12.2   C:\Users\Malith Manupriya\AppData\Local\Programs\Python\Python312\python.exe
numpy 2.4.6    scipy 1.15.3   gmsh 4.15.2    meshio 5.3.5
trimesh 5.1.0  rtree 1.4.1    pyvista 0.49.0 scikit-fem 12.0.2
scikit-image 0.26.0            fast-simplification 0.2.0   (Stage 7/8)
```

Reinstall with `pip install -r requirements.txt`.

Note: installing `scikit-image` for Stage 7 pulled numpy from 1.26.4 up to
2.4.6. Stages 3 and 5 were re-verified on numpy 2 and produce **identical**
numbers, so the upgrade is safe — but an unrelated `tensorflow-intel 2.16.1`
in this Python install pins `numpy<2.0` and will now warn. It is not used by
this project. Use a virtualenv if that matters.

## The pipeline

Eight stages, five entry scripts. Each stage is standalone and re-meshes from the
STEP file, so you can always jump straight to the one you care about — nothing
is cached between runs except what you explicitly `--save`.

```
config.py  ──►  main.py          Stage 1/2  pick + verify CAD faces
           ──►  voxel_main.py    Stage 3    CAD solid -> cubic voxel mesh -> BC node sets
           ──►  linear_check_main.py  Stage 4  independent linear FE sanity solve (scikit-fem)
           ──►  optimize_main.py  Stage 5/6  nonlinear constant-force topopt (top3d_cf.py)
           ──►  reconstruct_main.py Stage 7/8 density field -> STL -> STEP solid
```

### Stage 1/2 — face selection

```bash
python main.py --list      # print Face ID / area / centroid table
python main.py --show      # numbered 3-D CAD viewer (GUI)
python main.py --verify    # colour-coded CLAMP/INPUT/OUTPUT/DESIGN (GUI)
```

Read the integer Face IDs off `--show`, write them into `CLAMP` / `INPUT` /
`OUTPUT` in `config.py`, then confirm with `--verify`.

**Gmsh renumbers faces after any upstream SolidWorks edit.** Re-run `--list`
and re-check the IDs any time the STEP file changes.

### Stage 3 — voxel mesh

```bash
python voxel_main.py --verify            # build + 6 structural checks   <- the useful one
python voxel_main.py --show              # 3-D view of mesh + BC nodes (GUI)
python voxel_main.py --nel 40            # override grid resolution
python voxel_main.py --save mesh.npz     # write mesh + node sets out
```

The six checks catch: corner ordering vs `top3d_cf.py`'s reference cube,
degenerate elements, unsupported nodes (singular tangent), BC nodes that aren't
used+surface, a part severed into disconnected pieces by voxelization, and BC
sets straddling two components. If check 5 fails, a feature is thinner than one
voxel — raise `VOXEL_NEL_LONG_AXIS`.

### Stage 4 — linear check

```bash
python linear_check_main.py              # solve + report
python linear_check_main.py --show       # deformed shape (GUI)
```

Deliberately uses scikit-fem, **not** `top3d_cf.py`, so a mesh/BC wiring bug
can't be confirmed by the same code that caused it. Look for: clamp + input
reactions summing to zero, ~0 residual on free dofs, and nonzero output
displacement in the configured direction.

### Stage 5/6 — optimization

```bash
python optimize_main.py --verify                  # finite-difference self-test  <- run first
python optimize_main.py --time 5                  # time 5 iterations on the real part
python optimize_main.py --run                     # full run (maxiter from config.py)
python optimize_main.py --run --vtu design.vtu    # ... and export
```

`--verify` runs on a fixed small `cad_models/sample_bracket.step`, **not** your
config — so it stays meaningful while `config.py` is mid-edit. Run it after any
change to `optimization/voxel_model.py`.

**Always pass `--save-result`** on a real run:

```bash
python optimize_main.py --run --save-result output/result.npz
```

It stores the density field *together with the voxel grid*, which is what
Stages 7/8 need. Without it, re-tuning the CAD reconstruction means repeating
the whole optimisation. The `.vtu` cannot substitute — it holds only the active
hexahedra, and marching cubes needs the dense grid including empty cells.

### Stage 7/8 — CAD output (STL, STEP)

```bash
python reconstruct_main.py output/result.npz --stl design.stl
python reconstruct_main.py output/result.npz --stl d.stl --step d.step
python reconstruct_main.py output/result.npz --sigma 1.2 --decimate 3000 --stl d.stl
python reconstruct_main.py output/result.npz --show        # preview the surface
```

Or in one shot off a finished run (defaults only):

```bash
python optimize_main.py --run --save-result r.npz --stl d.stl --step d.step
```

Three knobs, all on `reconstruct_main.py`:

| Flag | Default | What it does |
|---|---|---|
| `--threshold` | 0.5 | Density level counted as material. Raise for a leaner part. |
| `--sigma` | 0.8 | Gaussian blur (in elements) applied **before** thresholding. This is what removes the voxel staircase. 0 = raw blocky surface; above ~1.5 starts eating thin members. |
| `--decimate` | 5000 | Target triangle count. Stage 8 emits **one CAD face per triangle** at ~2.4 kB each, so this decides whether the STEP opens comfortably. |

Stage 7 prints a 6-check PASS/FAIL report (watertight, winding, single body,
positive volume, volume vs. thresholded voxels, surface inside the grid). Stage
8 refuses to write a STEP if those fail — it cannot sew a solid from a surface
that isn't closed.

**What the STEP actually is:** a *faceted* B-Rep — one planar CAD face per
triangle, sewn into a single solid. It opens in SolidWorks as a real solid body
(measurable, meshable, usable for CAM), but it is **not** a parametric feature
tree with fillets and extrudes, and no automated step can make it one — an
organic optimised shape has no feature history to recover. Commercial tools
(nTopology, Inspire, Fusion) behave the same way.

**Dead end, already tested — do not retry:** Gmsh's `classifySurfaces()` +
`createGeometry()` (the STL-remeshing recipe from Gmsh's own tutorial) builds
*discrete* geometry, not OCC CAD data, and STEP export then fails outright with
`No suitable CAD data found for STEP export`. That is why
`cad_output/step_export.py` builds OCC points/lines/planar faces directly and
sews them, rather than using those calls.

## Last verified

All four stages run clean as of **2026-09-23**, on `cad_models/Part1.STEP` at
`VOXEL_NEL_LONG_AXIS = 24`.

| Stage | Command | Result |
|---|---|---|
| 1 | `main.py --list` | 15 faces found |
| 3 | `voxel_main.py --verify` | **PASS** — 6/6 checks |
| 4 | `linear_check_main.py` | equilibrium OK, residual 1.1e-11 |
| 5 | `optimize_main.py --verify` | **PASS** — fint 3.0e-09, Kt 1.3e-10, adjoint 1.0e-05 |
| 7 | `reconstruct_main.py` | **PASS** — watertight, 1 body, smooth (see below) |
| 8 | `reconstruct_main.py --step` | **PASS** — re-imports as 1 solid, volume err 1.8e-14 |

Stage 7/8 were verified two ways. First on a **synthetic all-solid density
field**, where the exact answer is known: the extracted surface reproduced the
voxel bounding box to **8.7e-19 m** and 99.2% of the exact volume at `--sigma 0`,
confirming the index-order and origin mapping are right. Then on a **real
3-iteration run** (`output/result.npz`): 2776 triangles, watertight, single body,
exported to a 6.7 MB STEP that re-imports as one solid with volume matching the
source mesh to 1.8e-14 relative error.

Stage 3 mesh, at `--nel 24`: grid 26x18x18, h = 6.25e-4 m, 3584 active elements
(42.5% fill), 4518 nodes / 13554 dof. Node sets: clamp 153, input 25, output 204.

Stage 4 linear solve: max |u| = 4.02e-3 m, mean output disp
[-1.30e-05, 0, 0] m, clamp reaction +7802 N / input reaction -7802 N (balanced).

## Runtime

Measured with `optimize_main.py --time 2` at `--nel 24` (29241 dof, 13070 free):

- iteration 1: 320 s (includes warm-up)
- iteration 2: 124 s

So **~2 min/iteration**. With `maxiter = 100` a full `--run` is **~3.5 hours**.

Stage 7/8 are cheap by comparison — seconds, which is the point of splitting
them off behind `--save-result`. STEP writing costs roughly **1.5 ms and 2.4 kB
per triangle**: 2776 triangles took a few seconds and produced 6.7 MB.
Cost scales steeply with `VOXEL_NEL_LONG_AXIS` — time a couple of iterations
before committing to a full run at a new resolution.

## Open issues

### 1. clamp/output node overlap on Part1.STEP — OPEN

Every stage prints:

```
note: 34 node(s) are in both 'clamp' and 'output'.
```

`CLAMP = [9]` (centroid x = -0.002) and `OUTPUT = [8, 10]` (x = -0.0035) share
an edge. Stage 5 currently resolves it by dropping those 34 dofs from the output
spring — harmless numerically (a Dirichlet dof carries no spring contribution),
but it means **the output spring is smaller than intended**, and the reported
output force is measured over a partly-clamped face.

Fix by choosing clamp and output faces that don't touch, before trusting any
`--run` result quantitatively.

### 2. INPUT and OUTPUT are on opposite ends — CHECK BEFORE A FULL RUN

`top3d_cf.py`'s README warns that input and output should sit on the **same free
end, along the same axis**, or the initial force path is sign-indefinite and the
optimiser refuses to start. Here `INPUT` is face 15 at x = +0.01 and `OUTPUT` is
faces 8/10 at x = -0.0035 — opposite ends. It did start (ripple was 124% → 112%
over two iterations), but this is worth confirming against the intended
mechanism before reading anything into a converged result.

### 3. `--threshold 0.5` only works on a *converged* run — by design

A SIMP/Heaviside field starts uniform at `volfrac` and only polarises towards
0/1 over many iterations. After a short run (`--time 3`) the whole field is
still grey — max density 0.314 — so nothing crosses 0.5 and Stage 7 refuses,
printing a volume-matched threshold to use instead (0.295 in that case).

That fallback gets you a **structurally valid** solid, and Stage 7 says so
explicitly while warning that it holds only 31% of the design's material. Do
not read a shape off a grey field: let the optimisation converge and keep
`--threshold 0.5`.

Note check 5 (volume fidelity) deliberately does **not** block the STEP export
— only checks 1-4 and 6 do, because those are the only ones that decide whether
a closed solid can be sewn. Fidelity is a judgement about the reconstruction,
not about geometric validity.

### 4. `config.STEP_FILE` case mismatch — cosmetic, Windows-only

`config.py` says `Part1.step`; the file on disk is `Part1.STEP`. Works on
Windows (case-insensitive FS), would break on Linux.

## Conventions

- `.gitignore` covers `__pycache__/`, `*.npz`, `*.png` — meshes and screenshots
  are regenerated, not committed.
- `output/` is currently empty; point `--vtu` / `--save` / `--screenshot` there.
- GUI flags (`--show`, `main.py --verify`) open a window and block. In a
  headless or agent session use `--screenshot FILE` instead and read the image.
