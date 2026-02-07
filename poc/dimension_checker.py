"""
Dimension verification checker for the PoC.

Compares extracted dimensions against target dimensions to ensure
the generated drawing matches the input specifications.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from poc.imperial_parser import ImperialDimension, parse as parse_imperial
from poc.pool_dxf_generator import PoolEdge


@dataclass
class DimensionCheck:
    """Result of checking a single dimension."""
    edge_index: int
    expected_inches: float
    actual_inches: float
    error_inches: float
    passed: bool
    label: str = ""


@dataclass
class VerificationResult:
    """Overall verification result."""
    checks: list[DimensionCheck] = field(default_factory=list)
    total_checked: int = 0
    total_passed: int = 0
    total_failed: int = 0
    all_passed: bool = False
    max_error_inches: float = 0.0

    @property
    def pass_rate(self) -> float:
        if self.total_checked == 0:
            return 0.0
        return self.total_passed / self.total_checked


def verify_dimensions(
    generated_edges: list[PoolEdge],
    target_dimensions: dict[int, float],
    tolerance_inches: float = 0.5,
) -> VerificationResult:
    """
    Verify that generated edge lengths match target dimensions.

    Args:
        generated_edges: List of edges from the generated drawing
        target_dimensions: Dict mapping edge_index -> expected length in inches
        tolerance_inches: Allowable error in inches (default ±0.5")

    Returns:
        VerificationResult with per-edge checks
    """
    result = VerificationResult()

    for edge_idx, expected in target_dimensions.items():
        if edge_idx >= len(generated_edges):
            continue

        edge = generated_edges[edge_idx]
        actual = edge.length
        error = abs(actual - expected)
        passed = error <= tolerance_inches

        check = DimensionCheck(
            edge_index=edge_idx,
            expected_inches=expected,
            actual_inches=round(actual, 2),
            error_inches=round(error, 2),
            passed=passed,
            label=edge.label or "",
        )
        result.checks.append(check)
        result.total_checked += 1
        if passed:
            result.total_passed += 1
        else:
            result.total_failed += 1
        result.max_error_inches = max(result.max_error_inches, error)

    result.all_passed = result.total_failed == 0 and result.total_checked > 0
    return result


def verify_bbox(
    edges: list[PoolEdge],
    target_length_inches: float,
    target_width_inches: float,
    tolerance_inches: float = 1.0,
) -> VerificationResult:
    """
    Verify generated edges against target dimensions using bounding box.

    Works for all pool types (rectangular, kidney, freeform, etc.)
    by comparing the overall bounding box against target length/width.
    """
    result = VerificationResult()

    if not edges:
        return result

    xs = [e.x1 for e in edges] + [e.x2 for e in edges]
    ys = [e.y1 for e in edges] + [e.y2 for e in edges]
    actual_length = max(xs) - min(xs)
    actual_width = max(ys) - min(ys)

    for label, expected, actual in [
        ("Length", target_length_inches, actual_length),
        ("Width", target_width_inches, actual_width),
    ]:
        error = abs(actual - expected)
        passed = error <= tolerance_inches
        check = DimensionCheck(
            edge_index=-1,
            expected_inches=expected,
            actual_inches=round(actual, 2),
            error_inches=round(error, 2),
            passed=passed,
            label=label,
        )
        result.checks.append(check)
        result.total_checked += 1
        if passed:
            result.total_passed += 1
        else:
            result.total_failed += 1
        result.max_error_inches = max(result.max_error_inches, error)

    result.all_passed = result.total_failed == 0 and result.total_checked > 0
    return result


def verify_from_ocr_results(
    generated_edges: list[PoolEdge],
    ocr_dimensions: list[ImperialDimension],
    tolerance_inches: float = 0.5,
) -> VerificationResult:
    """
    Verify generated edges against OCR-extracted dimensions.

    Matches each OCR dimension to the closest generated edge by length,
    then checks if they're within tolerance.
    """
    result = VerificationResult()

    # For each OCR dimension, find the closest matching edge
    used_edges = set()
    for dim in ocr_dimensions:
        best_idx = -1
        best_error = float("inf")

        for i, edge in enumerate(generated_edges):
            if i in used_edges:
                continue
            error = abs(edge.length - dim.total_inches)
            if error < best_error:
                best_error = error
                best_idx = i

        if best_idx >= 0:
            used_edges.add(best_idx)
            passed = best_error <= tolerance_inches
            check = DimensionCheck(
                edge_index=best_idx,
                expected_inches=dim.total_inches,
                actual_inches=round(generated_edges[best_idx].length, 2),
                error_inches=round(best_error, 2),
                passed=passed,
                label=dim.raw_text,
            )
            result.checks.append(check)
            result.total_checked += 1
            if passed:
                result.total_passed += 1
            else:
                result.total_failed += 1
            result.max_error_inches = max(result.max_error_inches, best_error)

    result.all_passed = result.total_failed == 0 and result.total_checked > 0
    return result
