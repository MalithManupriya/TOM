"""Stage 7/8: turn the optimised density field into a CAD deliverable.

Stage 7 (isosurface.py) extracts a smooth, watertight triangulated surface
from the voxel density field and writes it as STL -- the standard deliverable
for a topology-optimised part, and what goes to a printer.

Stage 8 (step_export.py) turns that surface into a real OpenCASCADE B-Rep
solid and writes it as STEP, for reopening in SolidWorks / further CAM.

Both are pure post-processing: they read the optimiser's output and never
touch top3d_cf.py or Stages 1-6.
"""
