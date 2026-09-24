"""
Turn the CLAMP / INPUT / OUTPUT / DESIGN_FACES lists from config.py into one
validated, non-overlapping face -> role assignment.

Kept deliberately dumb: no geometry here, just bookkeeping and validation, so
mistakes (typo'd face ID, a face assigned twice, a face nobody claimed) are
caught with a clear error message before you ever open the viewer or spend
time meshing.
"""
from __future__ import annotations

from dataclasses import dataclass

ROLES = ("clamp", "input", "output", "design")


@dataclass
class FaceAssignment:
    role_by_face: dict[int, str]  # face tag -> one of ROLES

    def faces_with_role(self, role: str) -> list[int]:
        return sorted(t for t, r in self.role_by_face.items() if r == role)


def resolve(all_face_tags: list[int], clamp: dict, input_: dict, output: dict,
            design_faces: list[int] | None) -> FaceAssignment:
    """Validate and merge the four face groups from config.py.

    Raises ValueError with a specific, actionable message on:
      - a face ID that does not exist on the imported CAD model
      - a face claimed by more than one role
      - a face left unclaimed by every role (only possible if DESIGN_FACES
        is given explicitly and misses one)
    """
    valid = set(all_face_tags)
    role_by_face: dict[int, str] = {}

    def claim(role: str, faces: list[int]) -> None:
        for tag in faces:
            if tag not in valid:
                raise ValueError(
                    f"{role.upper()}: face {tag} does not exist on this CAD "
                    f"model. Valid Face IDs are {sorted(valid)}. Run "
                    f"`python main.py --show` to re-check the numbering."
                )
            if tag in role_by_face:
                raise ValueError(
                    f"face {tag} is assigned to both '{role_by_face[tag]}' "
                    f"and '{role}' -- a face can only have one role."
                )
            role_by_face[tag] = role

    claim("clamp", clamp["faces"])
    claim("input", input_["faces"])
    claim("output", output["faces"])

    remaining = [t for t in all_face_tags if t not in role_by_face]
    design = design_faces if design_faces is not None else remaining
    claim("design", design)

    unassigned = [t for t in all_face_tags if t not in role_by_face]
    if unassigned:
        raise ValueError(
            f"faces {unassigned} are not assigned to any role. Add them to "
            f"DESIGN_FACES or to one of CLAMP/INPUT/OUTPUT in config.py."
        )

    return FaceAssignment(role_by_face=role_by_face)
