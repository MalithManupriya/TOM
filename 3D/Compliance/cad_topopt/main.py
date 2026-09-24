#!/usr/bin/env python3
"""
Stage 1 / Stage 2 entry point: CAD face identification and selection.

    python main.py --list      print every Face ID, area and centroid
    python main.py --show      open the numbered CAD viewer            (Stage 1)
    python main.py --verify    colour CLAMP/INPUT/OUTPUT/DESIGN and open
                                the viewer, so you can check the face
                                selection before doing anything else    (Stage 2)

Edit config.py to point at your STEP file and fill in the face IDs.
Once the selection looks right here, voxel_main.py turns it into a mesh.
"""
from __future__ import annotations

import argparse

import config
from cad.face_manager import resolve
from cad.import_cad import CadModel
from cad.viewer import show_assignment, show_face_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the face table")
    ap.add_argument("--show", action="store_true", help="numbered CAD viewer (Stage 1)")
    ap.add_argument("--verify", action="store_true",
                     help="colour the CLAMP/INPUT/OUTPUT/DESIGN assignment (Stage 2)")
    args = ap.parse_args()

    if not (args.list or args.show or args.verify):
        ap.print_help()
        return

    with CadModel(config.STEP_FILE) as cad:
        if args.list:
            cad.print_faces()

        if args.show:
            show_face_ids()

        if args.verify:
            assignment = resolve(cad.face_tags(), config.CLAMP, config.INPUT,
                                  config.OUTPUT, config.DESIGN_FACES)
            show_assignment(assignment)


if __name__ == "__main__":
    main()
