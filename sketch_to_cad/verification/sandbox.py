"""
Verification sandbox — iterative generate-verify-adjust loop.

Generates a DXF, verifies dimensions, and if they don't match,
computes correction factors and retries (up to 3 iterations).
"""

from dataclasses import dataclass, field
from typing import Optional

from sketch_to_cad.verification.dimension_compare import VerificationResult


@dataclass
class SandboxIteration:
    """Result of a single sandbox iteration."""
    iteration: int
    dxf_bytes: bytes
    verification: VerificationResult
    adjustments_made: dict
    passed: bool


@dataclass
class SandboxResult:
    """Result of the full sandbox run."""
    iterations: list[SandboxIteration] = field(default_factory=list)
    final_dxf_bytes: Optional[bytes] = None
    final_verification: Optional[VerificationResult] = None
    converged: bool = False
    total_iterations: int = 0


def run_sandbox(
    reference_dxf_path: str,
    target_length_inches: float,
    target_width_inches: float,
    max_iterations: int = 3,
    tolerance_inches: float = 0.5,
) -> SandboxResult:
    """
    Iterative generate -> verify -> adjust loop.

    Algorithm:
    1. deform_to_dimensions(ref, target_length, target_width)
    2. verify_bbox(result.edges, target_length, target_width)
    3. If passed -> done
    4. If failed -> compute correction factor:
       adj_length = target_length * (target_length / actual_length)
       adj_width  = target_width  * (target_width / actual_width)
    5. Retry with adjusted targets (up to max_iterations)
    """
    from sketch_to_cad.cad_engine.deformation import deform_to_dimensions
    from sketch_to_cad.verification.dimension_compare import verify_bbox

    result = SandboxResult()
    adj_length = target_length_inches
    adj_width = target_width_inches

    for i in range(max_iterations):
        deform = deform_to_dimensions(reference_dxf_path, adj_length, adj_width)
        verification = verify_bbox(
            deform.edges, target_length_inches, target_width_inches, tolerance_inches
        )

        iteration = SandboxIteration(
            iteration=i + 1,
            dxf_bytes=deform.dxf_bytes,
            verification=verification,
            adjustments_made={"length": round(adj_length, 2), "width": round(adj_width, 2)},
            passed=verification.all_passed,
        )
        result.iterations.append(iteration)

        if verification.all_passed:
            result.converged = True
            result.final_dxf_bytes = deform.dxf_bytes
            result.final_verification = verification
            break

        # Compute correction: if actual is X, target is Y, next try Y * (Y/X)
        for check in verification.checks:
            if check.label == "Length" and check.actual_inches > 0:
                adj_length *= target_length_inches / check.actual_inches
            elif check.label == "Width" and check.actual_inches > 0:
                adj_width *= target_width_inches / check.actual_inches

    if not result.converged and result.iterations:
        best = min(result.iterations, key=lambda it: it.verification.max_error_inches)
        result.final_dxf_bytes = best.dxf_bytes
        result.final_verification = best.verification

    result.total_iterations = len(result.iterations)
    return result
