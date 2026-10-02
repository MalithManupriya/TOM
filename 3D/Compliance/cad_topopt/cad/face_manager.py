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

_AXES = {"x": 0, "y": 1, "z": 2}
# the config key holding each grouped role's magnitude
_AMOUNT_KEY = {"input": "displacement", "output": "spring_stiffness"}


@dataclass
class FaceAssignment:
    role_by_face: dict[int, str]  # face tag -> one of ROLES

    def faces_with_role(self, role: str) -> list[int]:
        return sorted(t for t, r in self.role_by_face.items() if r == role)


@dataclass(frozen=True)
class BCGroup:
    """One INPUT or OUTPUT entry from config.py: some faces, one direction.

    amount : input  -> displacement along `direction`, m (so a signed
                       direction and a signed displacement multiply:
                       "x" with -0.004 and "-x" with 0.004 both push 4 mm
                       towards -x)
             output -> total spring stiffness of this group, N/m
    """
    role: str
    name: str
    faces: tuple[int, ...]
    axis: int       # 0, 1, 2 == x, y, z
    sign: float     # +1.0 or -1.0
    amount: float

    @property
    def direction(self) -> str:
        return ("+" if self.sign > 0 else "-") + "xyz"[self.axis]

    @property
    def prescribed(self) -> float:
        """Input only: the full-stroke displacement component along `axis`."""
        return self.sign * self.amount


def _entries(cfg: dict | list[dict]) -> list[dict]:
    """INPUT/OUTPUT may be one dict (a single group -- the original format)
    or a list of dicts (one group per face set / direction)."""
    return [cfg] if isinstance(cfg, dict) else list(cfg)


def parse_direction(text: str) -> tuple[int, float]:
    """'x', '+x' or '-x' (any axis, any case) -> (axis index, sign)."""
    s = str(text).strip().lower()
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-")
    if s not in _AXES:
        raise ValueError(f"direction {text!r} is not one of x, y, z "
                         f"(optionally prefixed with + or -).")
    return _AXES[s], sign


def role_faces(cfg: dict | list[dict]) -> list[int]:
    """Every face used by any group of one role."""
    return sorted({int(f) for e in _entries(cfg) for f in e["faces"]})


def bc_groups(role: str, cfg: dict | list[dict]) -> list[BCGroup]:
    """Parse and validate config.INPUT or config.OUTPUT into BCGroups."""
    key = _AMOUNT_KEY[role]
    entries = _entries(cfg)
    if not entries:
        raise ValueError(f"{role.upper()} has no groups -- at least one is needed.")

    groups = []
    for i, e in enumerate(entries):
        name = str(e.get("name", f"{role}[{i}]"))
        faces = tuple(int(f) for f in e.get("faces", []))
        if not faces:
            raise ValueError(f"{name}: 'faces' is empty.")
        for k in ("direction", key):
            if k not in e:
                raise ValueError(f"{name}: missing '{k}'.")
        axis, sign = parse_direction(e["direction"])
        amount = float(e[key])
        if role == "output" and amount <= 0:
            raise ValueError(f"{name}: spring_stiffness must be positive "
                             f"(got {amount}). The output direction is set "
                             f"by 'direction', not by the stiffness sign.")
        groups.append(BCGroup(role, name, faces, axis, sign, amount))
    return groups


def resolve(all_face_tags: list[int], clamp: dict, input_: dict | list[dict],
            output: dict | list[dict],
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

    # A face may appear in several groups of the SAME role (e.g. one input
    # face driven in both x and y), so each role claims its de-duplicated
    # face set once. Across roles, a face is still exclusive.
    claim("clamp", clamp["faces"])
    claim("input", role_faces(input_))
    claim("output", role_faces(output))

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
