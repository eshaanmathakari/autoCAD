"""
Pool dimension validation against building codes.

Checks dimensions against industry standards and flags issues.
"""

from dataclasses import dataclass

from sketch_to_cad.domain.constants import (
    POOL_LENGTH_RANGE,
    POOL_WIDTH_RANGE,
    POOL_DEPTH_RANGE,
    MAX_RISER_HEIGHT_RESIDENTIAL,
    MIN_TREAD_DEPTH,
    MIN_TREAD_AREA,
    SAFETY_LEDGE_WIDTH_RANGE,
)


@dataclass
class ValidationIssue:
    """A single validation finding."""
    severity: str   # "error" | "warning"
    code: str       # e.g. "DEPTH_BELOW_MIN"
    message: str
    field: str


def validate_pool_dimensions(
    length_inches: float,
    width_inches: float,
    shallow_depth_inches: float = 42,
    deep_depth_inches: float = 96,
) -> list[ValidationIssue]:
    """Check pool dimensions against building codes and plausible ranges."""
    issues: list[ValidationIssue] = []

    lo, hi = POOL_LENGTH_RANGE
    if not (lo <= length_inches <= hi):
        issues.append(ValidationIssue(
            severity="warning",
            code="LENGTH_OUT_OF_RANGE",
            message=f"Pool length {length_inches:.0f}\" is outside typical range ({lo}\"-{hi}\")",
            field="length",
        ))

    lo, hi = POOL_WIDTH_RANGE
    if not (lo <= width_inches <= hi):
        issues.append(ValidationIssue(
            severity="warning",
            code="WIDTH_OUT_OF_RANGE",
            message=f"Pool width {width_inches:.0f}\" is outside typical range ({lo}\"-{hi}\")",
            field="width",
        ))

    lo, hi = POOL_DEPTH_RANGE
    if not (lo <= shallow_depth_inches <= hi):
        issues.append(ValidationIssue(
            severity="error",
            code="SHALLOW_DEPTH_OUT_OF_RANGE",
            message=f"Shallow depth {shallow_depth_inches:.0f}\" is outside range ({lo}\"-{hi}\")",
            field="shallow_depth",
        ))
    if not (lo <= deep_depth_inches <= hi):
        issues.append(ValidationIssue(
            severity="error",
            code="DEEP_DEPTH_OUT_OF_RANGE",
            message=f"Deep depth {deep_depth_inches:.0f}\" is outside range ({lo}\"-{hi}\")",
            field="deep_depth",
        ))

    if shallow_depth_inches > deep_depth_inches:
        issues.append(ValidationIssue(
            severity="error",
            code="DEPTH_INVERTED",
            message="Shallow depth exceeds deep depth",
            field="shallow_depth",
        ))

    return issues


def validate_stair_spec(
    width_inches: float,
    depth_inches: float,
    num_treads: int,
) -> list[ValidationIssue]:
    """Check stair dimensions against pool codes."""
    issues: list[ValidationIssue] = []

    if num_treads < 1:
        issues.append(ValidationIssue(
            severity="error",
            code="NO_TREADS",
            message="Stairs must have at least 1 tread",
            field="num_treads",
        ))
        return issues

    riser_height = depth_inches / num_treads
    if riser_height > MAX_RISER_HEIGHT_RESIDENTIAL:
        issues.append(ValidationIssue(
            severity="error",
            code="RISER_TOO_HIGH",
            message=f"Riser height {riser_height:.1f}\" exceeds max {MAX_RISER_HEIGHT_RESIDENTIAL}\"",
            field="num_treads",
        ))

    tread_depth = depth_inches / num_treads
    if tread_depth < MIN_TREAD_DEPTH:
        issues.append(ValidationIssue(
            severity="warning",
            code="TREAD_TOO_SHALLOW",
            message=f"Tread depth {tread_depth:.1f}\" below minimum {MIN_TREAD_DEPTH}\"",
            field="depth",
        ))

    tread_area = width_inches * tread_depth
    if tread_area < MIN_TREAD_AREA:
        issues.append(ValidationIssue(
            severity="warning",
            code="TREAD_AREA_TOO_SMALL",
            message=f"Tread area {tread_area:.0f} sq in below minimum {MIN_TREAD_AREA} sq in",
            field="width",
        ))

    return issues
