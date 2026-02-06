"""
Scale Calibration Module for CAD Sketch Extraction.

Provides pixel-to-mm calibration using:
1. Scale notation parsing (e.g., "1:100", "1:50")
2. Dimension annotation analysis
3. Reference object detection (ruler bars)
"""

import re
import math
from typing import Optional
from PIL import Image
import numpy as np
import cv2
from pydantic import BaseModel, Field

from .ocr_utils import OCRCorrector


class CalibrationResult(BaseModel):
    """Result of scale calibration."""
    pixels_per_mm: float = Field(..., description="Pixel to mm conversion ratio")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibration confidence")
    method: str = Field(..., description="Calibration method used")
    reference_description: str = Field(..., description="What was used for calibration")
    scale_factor: float = Field(default=1.0, description="Drawing scale factor (e.g., 100 for 1:100)")


class ScaleCalibrator:
    """
    Detects and calibrates pixel-to-mm ratio from sketches.

    Supports multiple calibration strategies:
    1. Scale notation parsing (e.g., "1:100", "Scale 1:50")
    2. Dimension-based inference from annotated measurements
    3. Ruler/scale bar detection (experimental)

    Priority order: dimension analysis > scale notation > ruler detection
    """

    # Common paper sizes in mm (width x height in landscape)
    PAPER_SIZES = {
        'A4': (297, 210),
        'A3': (420, 297),
        'A2': (594, 420),
        'A1': (841, 594),
        'A0': (1189, 841),
        'LETTER': (279.4, 215.9),
        'LEGAL': (355.6, 215.9),
        'TABLOID': (431.8, 279.4),
    }

    # Common scale factors
    COMMON_SCALES = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]

    def __init__(
        self,
        image: Image.Image,
        assumed_paper: str = 'A4'
    ):
        """
        Initialize calibrator with image.

        Args:
            image: PIL Image to calibrate
            assumed_paper: Assumed paper size for fallback calculations
        """
        self.image = image
        self.width, self.height = image.size
        self.assumed_paper = assumed_paper.upper()
        self.cv_image = np.array(image.convert('RGB'))

    def detect_scale_notation(
        self,
        metadata_scale: Optional[str]
    ) -> Optional[CalibrationResult]:
        """
        Parse scale notation like "1:100" or "1:50".

        Uses the assumed paper size to compute pixels_per_mm from scale.

        Args:
            metadata_scale: Scale string from Gemini extraction
                           (e.g., "1:100", "Scale: 1/50", "1-100")

        Returns:
            CalibrationResult if scale found and parseable, None otherwise
        """
        if not metadata_scale:
            return None

        # Clean up the string
        scale_str = metadata_scale.strip().upper()

        # Try various patterns
        patterns = [
            r'1\s*[:/\-]\s*(\d+)',           # 1:100, 1/100, 1-100
            r'SCALE\s*[:\s]*1\s*[:/\-]\s*(\d+)',  # Scale: 1:100
            r'(\d+)\s*[:/\-]\s*1\b',          # 100:1 (inverse notation)
        ]

        scale_factor = None
        for pattern in patterns:
            match = re.search(pattern, scale_str)
            if match:
                scale_factor = int(match.group(1))
                break

        if scale_factor is None:
            return None

        # Get paper dimensions
        paper_dims = self.PAPER_SIZES.get(self.assumed_paper, self.PAPER_SIZES['A4'])

        # Determine if image is portrait or landscape
        if self.width > self.height:
            paper_width_mm, paper_height_mm = paper_dims
        else:
            paper_height_mm, paper_width_mm = paper_dims

        # Calculate pixels per mm of the actual paper
        # At scale 1:100, 1mm on paper represents 100mm in reality
        # So for the DXF output, we need to know the real-world mm per pixel
        pixels_per_paper_mm = self.width / paper_width_mm

        # Real-world mm = paper mm * scale_factor
        # So pixels per real-world mm = pixels / (paper_mm * scale_factor)
        pixels_per_real_mm = pixels_per_paper_mm / scale_factor

        return CalibrationResult(
            pixels_per_mm=pixels_per_real_mm,
            confidence=0.7,  # Moderate confidence (assumed paper size)
            method="scale_notation",
            reference_description=f"Scale {metadata_scale}, assumed {self.assumed_paper} paper",
            scale_factor=float(scale_factor)
        )

    def calibrate_from_dimensions(
        self,
        dimension_entities: list,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ) -> Optional[CalibrationResult]:
        """
        Infer scale from dimension annotations and their pixel positions.

        Strategy:
        1. For each dimension with start/end points and value
        2. Calculate pixel distance between points
        3. Compute pixels_per_mm = pixel_distance / dimension_value_mm
        4. Use median across all dimensions for robustness

        Args:
            dimension_entities: List of DimensionEntity objects
            image_width: Override image width (for normalized coord conversion)
            image_height: Override image height

        Returns:
            CalibrationResult based on dimension analysis, or None
        """
        if not dimension_entities:
            return None

        img_w = image_width or self.width
        img_h = image_height or self.height

        pixel_ratios = []

        for dim in dimension_entities:
            # Get dimension value
            value_text = getattr(dim, 'value', None)
            if not value_text:
                continue

            # Parse value
            value, unit = OCRCorrector.extract_numeric_value(str(value_text))
            if value is None or value <= 0:
                continue

            # Convert to mm
            value_mm = OCRCorrector.convert_to_mm(value, unit or 'mm')

            # Get pixel distance from normalized coordinates
            start = getattr(dim, 'start', None)
            end = getattr(dim, 'end', None)

            if not start or not end:
                continue

            # Convert normalized (0-1000) to pixels
            dx = (end.x - start.x) / 1000.0 * img_w
            dy = (end.y - start.y) / 1000.0 * img_h
            pixel_dist = math.sqrt(dx * dx + dy * dy)

            # Minimum distance threshold
            if pixel_dist < 20:
                continue

            ratio = pixel_dist / value_mm
            pixel_ratios.append({
                'ratio': ratio,
                'value_mm': value_mm,
                'pixel_dist': pixel_dist,
                'dimension': value_text
            })

        if not pixel_ratios:
            return None

        # Use median for robustness against outliers
        ratios_sorted = sorted(pixel_ratios, key=lambda r: r['ratio'])
        median_idx = len(ratios_sorted) // 2
        median_ratio = ratios_sorted[median_idx]['ratio']

        # Calculate standard deviation for confidence
        if len(pixel_ratios) >= 3:
            mean_ratio = sum(r['ratio'] for r in pixel_ratios) / len(pixel_ratios)
            variance = sum((r['ratio'] - mean_ratio) ** 2 for r in pixel_ratios) / len(pixel_ratios)
            std_ratio = math.sqrt(variance)
            cv = std_ratio / mean_ratio if mean_ratio > 0 else 1.0

            # Confidence based on consistency
            if cv < 0.1:
                confidence = 0.95
            elif cv < 0.2:
                confidence = 0.85
            elif cv < 0.3:
                confidence = 0.7
            else:
                confidence = 0.5
        else:
            confidence = 0.6

        # Try to infer scale factor
        # If we know pixels_per_mm and assume A4, we can estimate scale
        paper_width_mm = self.PAPER_SIZES.get(self.assumed_paper, (297, 210))[0]
        implied_real_width_mm = img_w / median_ratio
        implied_scale = implied_real_width_mm / paper_width_mm

        # Find closest common scale
        closest_scale = min(
            self.COMMON_SCALES,
            key=lambda s: abs(s - implied_scale)
        )

        return CalibrationResult(
            pixels_per_mm=median_ratio,
            confidence=confidence,
            method="dimension_analysis",
            reference_description=f"Derived from {len(pixel_ratios)} dimension annotations",
            scale_factor=float(closest_scale)
        )

    def detect_ruler_bar(self) -> Optional[CalibrationResult]:
        """
        Detect a scale bar/ruler in the image.

        Uses Hough line detection to find horizontal lines with
        evenly spaced tick marks, which might indicate a scale bar.

        Returns:
            CalibrationResult if ruler detected, None otherwise

        Note: This is experimental and may have lower accuracy.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None:
            return None

        # Find horizontal lines (within 5 degrees of horizontal)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)

            if angle < 5 or angle > 175:
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                horizontal_lines.append({
                    'line': (x1, y1, x2, y2),
                    'length': length,
                    'y': (y1 + y2) / 2
                })

        if not horizontal_lines:
            return None

        # Look for scale bar characteristics:
        # - Long horizontal line near top or bottom
        # - Has vertical tick marks nearby

        # Sort by length (longest first)
        horizontal_lines.sort(key=lambda l: l['length'], reverse=True)

        # For now, return None - full ruler detection requires more sophisticated
        # analysis including tick mark detection and OCR of ruler labels
        # This would be a future enhancement

        return None

    def calibrate(
        self,
        metadata_scale: Optional[str] = None,
        dimension_entities: Optional[list] = None,
        geometry_entities: Optional[list] = None
    ) -> Optional[CalibrationResult]:
        """
        Run calibration using the best available method.

        Priority order:
        1. Dimension-based calibration (most accurate when available)
        2. Scale notation parsing
        3. Ruler detection (experimental)

        Args:
            metadata_scale: Scale notation from extraction (e.g., "1:100")
            dimension_entities: List of DimensionEntity objects
            geometry_entities: List of all geometry entities (for future use)

        Returns:
            CalibrationResult from best method, or None if calibration fails
        """
        results = []

        # Try dimension-based calibration first
        if dimension_entities:
            result = self.calibrate_from_dimensions(dimension_entities)
            if result and result.confidence >= 0.5:
                results.append(result)

        # Try scale notation
        if metadata_scale:
            result = self.detect_scale_notation(metadata_scale)
            if result:
                results.append(result)

        # Try ruler detection (experimental)
        result = self.detect_ruler_bar()
        if result:
            results.append(result)

        if not results:
            return None

        # Return highest confidence result
        return max(results, key=lambda r: r.confidence)

    def apply_calibration_to_scale(
        self,
        calibration: CalibrationResult,
        user_scale: float = 1.0
    ) -> float:
        """
        Calculate the final scale factor to apply to DXF output.

        Args:
            calibration: Calibration result
            user_scale: User-specified additional scale factor

        Returns:
            Combined scale factor for DXF synthesis
        """
        # The base scale converts normalized coordinates to real-world mm
        # Then we apply the user's additional scale preference
        return user_scale / calibration.scale_factor


def calibrate_from_extraction(
    image: Image.Image,
    metadata_scale: Optional[str],
    dimension_entities: list,
    assumed_paper: str = 'A4'
) -> Optional[CalibrationResult]:
    """
    Convenience function to calibrate from extraction results.

    Args:
        image: Original image
        metadata_scale: Scale notation from Gemini
        dimension_entities: Dimension entities from extraction
        assumed_paper: Assumed paper size

    Returns:
        CalibrationResult or None
    """
    calibrator = ScaleCalibrator(image, assumed_paper)
    return calibrator.calibrate(
        metadata_scale=metadata_scale,
        dimension_entities=dimension_entities
    )
