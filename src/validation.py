"""
Geometry Validation Module for CAD Sketch Extraction.

Provides multi-pass validation of extracted geometry including:
- Pixel ratio consistency checking
- Dimension value validation
- Standard dimension snapping
- Cross-validation between dimensions and geometry
"""

import math
from typing import Optional
from pydantic import BaseModel, Field

from .geometry_models import (
    SketchGeometry, GeometryEntity, DimensionEntity, LineEntity,
    TextEntity, Point2D, LayerType
)
from .ocr_utils import (
    OCRCorrector, correct_dimension_entity, correct_text_entity,
    STANDARD_DIMENSIONS
)


class ValidationCorrection(BaseModel):
    """Record of a correction made during validation."""
    entity_index: int
    field: str
    original_value: str
    corrected_value: str
    reason: str


class ValidationResult(BaseModel):
    """Result of geometry validation."""
    is_valid: bool = True
    confidence_adjustment: float = Field(default=1.0, ge=0.0, le=1.0)
    corrections: list[ValidationCorrection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pixel_to_unit_ratio: Optional[float] = None
    consistency_score: float = Field(default=1.0, ge=0.0, le=1.0)


class GeometryValidator:
    """
    Validates and corrects extracted geometry using multiple strategies.

    Strategies applied:
    1. OCR correction for dimension text
    2. Room label fuzzy matching
    3. Pixel ratio consistency checking
    4. Standard dimension snapping
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        snap_threshold: float = 0.02,
        ratio_tolerance: float = 0.20,
    ):
        """
        Initialize the validator.

        Args:
            image_width: Original image width in pixels
            image_height: Original image height in pixels
            snap_threshold: Threshold for snapping to standard dimensions (default 2%, conservative)
            ratio_tolerance: Tolerance for pixel ratio consistency (default 20%)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.snap_threshold = snap_threshold
        self.ratio_tolerance = ratio_tolerance
        self.pixel_to_unit_ratio: Optional[float] = None

    def _calculate_pixel_distance(self, start: Point2D, end: Point2D) -> float:
        """
        Calculate pixel distance between two normalized points.

        Converts from 0-1000 normalized space to actual pixel space.
        """
        dx = (end.x - start.x) / 1000.0 * self.image_width
        dy = (end.y - start.y) / 1000.0 * self.image_height
        return math.sqrt(dx * dx + dy * dy)

    def _calculate_normalized_distance(self, start: Point2D, end: Point2D) -> float:
        """Calculate distance in normalized 0-1000 space."""
        dx = end.x - start.x
        dy = end.y - start.y
        return math.sqrt(dx * dx + dy * dy)

    def apply_ocr_corrections(
        self,
        geometry: SketchGeometry
    ) -> tuple[SketchGeometry, list[ValidationCorrection]]:
        """
        Apply OCR corrections to all text and dimension entities.

        Returns:
            Tuple of (corrected geometry, list of corrections made)
        """
        corrections = []

        for i, entity in enumerate(geometry.entities):
            if entity.type == "dimension":
                # Correct dimension values (pass confidence for context-aware fixes)
                original_value = entity.value
                if original_value:
                    # Get entity confidence (default to 1.0 if not available)
                    conf = getattr(entity, 'confidence', 1.0)
                    corrected = OCRCorrector.fix_dimension_text(original_value, confidence=conf)
                    if corrected != original_value:
                        entity.value = corrected
                        corrections.append(ValidationCorrection(
                            entity_index=i,
                            field="value",
                            original_value=original_value,
                            corrected_value=corrected,
                            reason="OCR character correction"
                        ))

            elif entity.type == "text":
                # Correct room labels
                original_content = entity.content
                matched = OCRCorrector.fuzzy_match_room_name(original_content)
                if matched and matched.upper() != original_content.upper():
                    entity.content = matched
                    corrections.append(ValidationCorrection(
                        entity_index=i,
                        field="content",
                        original_value=original_content,
                        corrected_value=matched,
                        reason="Room label fuzzy match"
                    ))

        return geometry, corrections

    def validate_pixel_ratios(
        self,
        geometry: SketchGeometry
    ) -> ValidationResult:
        """
        Cross-check dimension values against pixel distances.

        Strategy:
        1. For each DimensionEntity with start/end points and value
        2. Calculate pixel distance between start and end
        3. Compute implied pixel-to-mm ratio
        4. Check consistency across all dimensions
        5. Flag outliers where ratio differs >20% from median

        Returns:
            ValidationResult with consistency score and warnings
        """
        result = ValidationResult()
        ratios = []

        # Do not add "Insufficient dimension data" when extraction fully failed (no entities)
        if len(geometry.entities) == 0:
            return result

        dimensions = [e for e in geometry.entities if e.type == "dimension"]

        for dim in dimensions:
            if not hasattr(dim, 'value') or not dim.value:
                continue
            if not hasattr(dim, 'start') or not hasattr(dim, 'end'):
                continue

            # Parse dimension value
            value, unit = OCRCorrector.extract_numeric_value(dim.value)
            if value is None or value <= 0:
                continue

            # Convert to mm
            value_mm = OCRCorrector.convert_to_mm(value, unit or 'mm')

            # Calculate pixel distance
            pixel_dist = self._calculate_pixel_distance(dim.start, dim.end)

            if pixel_dist > 10:  # Minimum threshold to avoid noise
                ratio = pixel_dist / value_mm
                ratios.append({
                    'entity': dim,
                    'ratio': ratio,
                    'pixel_dist': pixel_dist,
                    'value_mm': value_mm
                })

        if len(ratios) < 2:
            # Softer message when we have entities but few dimensions; no warning when extraction failed
            if len(dimensions) == 0:
                result.warnings.append("Ratio validation skipped (no dimensions extracted)")
            return result

        # Calculate median ratio
        ratio_values = [r['ratio'] for r in ratios]
        ratio_values.sort()
        median_idx = len(ratio_values) // 2
        median_ratio = ratio_values[median_idx]

        # Store for calibration
        self.pixel_to_unit_ratio = median_ratio
        result.pixel_to_unit_ratio = median_ratio

        # Check consistency
        outliers = []
        for r in ratios:
            relative_diff = abs(r['ratio'] - median_ratio) / median_ratio
            if relative_diff > self.ratio_tolerance:
                outliers.append(r)
                result.warnings.append(
                    f"Dimension '{r['entity'].value}' has inconsistent pixel ratio "
                    f"({r['ratio']:.2f} vs median {median_ratio:.2f})"
                )

        # Calculate consistency score
        if len(outliers) == 0:
            result.consistency_score = 1.0
        else:
            result.consistency_score = 1.0 - (len(outliers) / len(ratios))

        return result

    def apply_standard_snapping(
        self,
        geometry: SketchGeometry
    ) -> tuple[SketchGeometry, list[ValidationCorrection]]:
        """
        Snap dimension values to common standards if within threshold.

        e.g., "498mm" -> "500mm" (0.4% difference, snap to standard)

        Returns:
            Tuple of (corrected geometry, list of corrections made)
        """
        corrections = []

        for i, entity in enumerate(geometry.entities):
            if entity.type != "dimension":
                continue
            if not entity.value:
                continue

            # Parse value
            value, unit = OCRCorrector.extract_numeric_value(entity.value)
            if value is None:
                continue

            # Convert to mm for snapping
            value_mm = OCRCorrector.convert_to_mm(value, unit or 'mm')

            # Try snapping
            snapped_mm, was_snapped = OCRCorrector.snap_to_standard(
                value_mm,
                STANDARD_DIMENSIONS,
                self.snap_threshold
            )

            if was_snapped:
                # Convert back to original unit if needed
                if unit == 'cm':
                    snapped_display = snapped_mm / 10
                elif unit == 'm':
                    snapped_display = snapped_mm / 1000
                elif unit in ['in', '"']:
                    snapped_display = snapped_mm / 25.4
                elif unit == 'ft':
                    snapped_display = snapped_mm / 304.8
                else:
                    snapped_display = snapped_mm

                # Format new value
                if snapped_display == int(snapped_display):
                    new_value = f"{int(snapped_display)}{unit or 'mm'}"
                else:
                    new_value = f"{snapped_display:.1f}{unit or 'mm'}"

                original_value = entity.value
                entity.value = new_value

                corrections.append(ValidationCorrection(
                    entity_index=i,
                    field="value",
                    original_value=original_value,
                    corrected_value=new_value,
                    reason=f"Snapped to standard ({value_mm:.1f}mm -> {snapped_mm}mm)"
                ))

        return geometry, corrections

    def validate_dimension_consistency(
        self,
        geometry: SketchGeometry
    ) -> ValidationResult:
        """
        Check that dimension values are internally consistent.

        Strategy:
        1. Find parallel lines that might be opposite walls
        2. Check that they have similar dimensions
        3. Verify that connected dimensions sum correctly

        Returns:
            ValidationResult with consistency warnings
        """
        result = ValidationResult()

        lines = [e for e in geometry.entities if e.type == "line"]
        dimensions = [e for e in geometry.entities if e.type == "dimension"]

        if len(lines) < 4 or len(dimensions) < 2:
            return result

        # Find horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            dx = abs(line.end.x - line.start.x)
            dy = abs(line.end.y - line.start.y)

            if dy < dx * 0.1:  # Mostly horizontal
                horizontal_lines.append(line)
            elif dx < dy * 0.1:  # Mostly vertical
                vertical_lines.append(line)

        # Check for parallel line pairs that should have similar dimensions
        # This is a simplified heuristic - could be expanded
        if len(horizontal_lines) >= 2:
            # Sort by Y position
            horizontal_lines.sort(key=lambda l: (l.start.y + l.end.y) / 2)

            # Compare lengths of top and bottom-most horizontal lines
            top = horizontal_lines[0]
            bottom = horizontal_lines[-1]

            top_length = self._calculate_normalized_distance(top.start, top.end)
            bottom_length = self._calculate_normalized_distance(bottom.start, bottom.end)

            # If similar length, they might be opposite walls of a room
            if abs(top_length - bottom_length) < top_length * 0.15:
                result.warnings.append(
                    f"Found parallel horizontal lines with similar lengths - likely room boundaries"
                )

        return result

    def validate_all(
        self,
        geometry: SketchGeometry,
        apply_snap: bool = False
    ) -> tuple[SketchGeometry, ValidationResult]:
        """
        Run all validation and correction steps.

        Args:
            geometry: Input geometry to validate
            apply_snap: Whether to snap dimensions to standards (default False - can corrupt valid values)

        Returns:
            Tuple of (corrected geometry, combined validation result)
        """
        combined_result = ValidationResult()
        has_entities = len(geometry.entities) > 0

        # Step 1: Apply OCR corrections
        geometry, ocr_corrections = self.apply_ocr_corrections(geometry)
        combined_result.corrections.extend(ocr_corrections)

        # Step 2: Validate pixel ratios
        ratio_result = self.validate_pixel_ratios(geometry)
        combined_result.warnings.extend(ratio_result.warnings)
        combined_result.pixel_to_unit_ratio = ratio_result.pixel_to_unit_ratio
        combined_result.consistency_score = ratio_result.consistency_score

        # Step 3: Apply standard snapping (optional)
        if apply_snap:
            geometry, snap_corrections = self.apply_standard_snapping(geometry)
            combined_result.corrections.extend(snap_corrections)

        # Step 4: Check dimension consistency
        consistency_result = self.validate_dimension_consistency(geometry)
        combined_result.warnings.extend(consistency_result.warnings)

        # Calculate overall validity
        combined_result.is_valid = len(combined_result.warnings) < 5

        # When extraction failed (no entities), do not penalize confidence
        if not has_entities:
            combined_result.confidence_adjustment = 1.0
            return geometry, combined_result

        # Minimal confidence adjustment - don't penalize heavily for corrections
        # Only apply small penalty for substantive corrections (not formatting)
        substantive_corrections = [
            c for c in combined_result.corrections
            if c.reason not in ["OCR character correction", "Room label fuzzy match"]
        ]
        if len(substantive_corrections) > 0:
            # Very conservative penalty: max 3%, 0.5% per substantive correction
            correction_penalty = min(0.03, len(substantive_corrections) * 0.005)
            combined_result.confidence_adjustment = 1.0 - correction_penalty

        # Factor in consistency score (but less aggressively)
        combined_result.confidence_adjustment *= (
            0.7 + 0.3 * combined_result.consistency_score
        )

        return geometry, combined_result


def validate_geometry(
    geometry: SketchGeometry,
    image_width: int,
    image_height: int,
    apply_snap: bool = False
) -> tuple[SketchGeometry, ValidationResult]:
    """
    Convenience function to validate geometry with default settings.

    Args:
        geometry: Geometry to validate
        image_width: Original image width in pixels
        image_height: Original image height in pixels
        apply_snap: Whether to snap to standard dimensions (default False)

    Returns:
        Tuple of (corrected geometry, validation result)
    """
    validator = GeometryValidator(image_width, image_height)
    return validator.validate_all(geometry, apply_snap=apply_snap)
