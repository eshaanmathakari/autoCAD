"""
PaddleOCR Integration for CAD Sketch Extraction.

Provides specialized OCR for dimension and label text using PaddleOCR,
which excels at handwritten text, rotated/angled text, and mixed
numeric/text content.

PaddleOCR 3.0/PP-OCRv5 features:
- 13-point accuracy gain for handwriting over PP-OCRv4
- State-of-the-art for cursive scripts and rotated text
- Supports 100+ languages
"""

import logging
from typing import Optional
from PIL import Image
import numpy as np
from pydantic import BaseModel, Field

from .ocr_utils import OCRCorrector, STANDARD_ROOM_NAMES


logger = logging.getLogger(__name__)


class OCRResult(BaseModel):
    """Single OCR detection result."""
    text: str = Field(..., description="Detected text content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: list[tuple[int, int]] = Field(..., description="Bounding box (4 corner points)")
    center_x: int = Field(..., description="Center X coordinate in pixels")
    center_y: int = Field(..., description="Center Y coordinate in pixels")
    width: int = Field(default=0, description="Bounding box width")
    height: int = Field(default=0, description="Bounding box height")
    rotation: float = Field(default=0.0, description="Estimated rotation angle in degrees")


class OCRExtractor:
    """
    Specialized OCR for dimension and label text using PaddleOCR.

    PaddleOCR 3.0/PP-OCRv5 excels at:
    - Handwritten text recognition
    - Rotated/angled text detection
    - Mixed numeric/text content (e.g., "25.5mm", "R15")
    - Technical drawing annotations
    """

    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        """
        Initialize PaddleOCR.

        Args:
            lang: Language code ('en' for English, 'ch' for Chinese, etc.)
            use_gpu: Whether to use GPU acceleration

        Note:
            PaddleOCR is an optional dependency. If not installed,
            the extractor will be disabled and methods will return empty results.
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr = None
        self.available = False

        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(
                use_angle_cls=True,     # Detect text rotation
                lang=lang,
                show_log=False,         # Reduce console spam
                use_gpu=use_gpu,
                det_model_dir=None,     # Use default PP-OCRv5 detection
                rec_model_dir=None,     # Use default PP-OCRv5 recognition
                cls_model_dir=None,     # Use default angle classifier
            )
            self.available = True
            logger.info("PaddleOCR initialized successfully")

        except ImportError:
            logger.warning(
                "PaddleOCR not installed. Install with: "
                "pip install paddlepaddle paddleocr"
            )
            self.available = False

        except RuntimeError as e:
            # Handle "PDX has already been initialized" error
            if "already been initialized" in str(e):
                logger.warning("PaddleOCR initialization conflict - disabling for this session")
            else:
                logger.warning(f"PaddleOCR runtime error: {e}")
            self.available = False

        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            self.available = False

    def extract_text_regions(self, image: Image.Image) -> list[OCRResult]:
        """
        Extract all text regions from image.

        Args:
            image: PIL Image to process

        Returns:
            List of OCRResult with text, confidence, and bounding boxes
        """
        if not self.available or self.ocr is None:
            return []

        try:
            # Convert PIL to numpy array (RGB)
            img_array = np.array(image.convert('RGB'))

            # Run OCR
            results = self.ocr.ocr(img_array, cls=True)

            ocr_results = []

            if results and results[0]:
                for line in results[0]:
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text, confidence = line[1]

                    # Calculate center point
                    center_x = int(sum(p[0] for p in bbox) / 4)
                    center_y = int(sum(p[1] for p in bbox) / 4)

                    # Calculate width and height
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    width = int(max(xs) - min(xs))
                    height = int(max(ys) - min(ys))

                    # Estimate rotation from bounding box
                    # (angle between top edge and horizontal)
                    dx = bbox[1][0] - bbox[0][0]
                    dy = bbox[1][1] - bbox[0][1]
                    rotation = np.arctan2(dy, dx) * 180 / np.pi

                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=float(confidence),
                        bbox=[(int(p[0]), int(p[1])) for p in bbox],
                        center_x=center_x,
                        center_y=center_y,
                        width=width,
                        height=height,
                        rotation=rotation
                    ))

            return ocr_results

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []

    def extract_dimensions(self, image: Image.Image) -> list[OCRResult]:
        """
        Extract only dimension-like text (numbers with optional units).

        Filters OCR results to keep only text that appears to be
        dimensional annotations (e.g., "500mm", "25.5", "R15").

        Args:
            image: PIL Image to process

        Returns:
            List of OCRResult containing dimension text only
        """
        all_text = self.extract_text_regions(image)
        dimensions = []

        for result in all_text:
            # Apply OCR corrections
            corrected = OCRCorrector.fix_dimension_text(result.text)
            value, unit = OCRCorrector.extract_numeric_value(corrected)

            if value is not None:
                # Update text with corrected version
                result.text = corrected
                dimensions.append(result)

        return dimensions

    def extract_room_labels(self, image: Image.Image) -> list[OCRResult]:
        """
        Extract room label text (alphabetic content matching room names).

        Args:
            image: PIL Image to process

        Returns:
            List of OCRResult containing room labels only
        """
        all_text = self.extract_text_regions(image)
        labels = []

        for result in all_text:
            # Check if it matches a room name pattern
            matched = OCRCorrector.fuzzy_match_room_name(
                result.text,
                STANDARD_ROOM_NAMES
            )

            if matched:
                # Use corrected name
                result.text = matched
                labels.append(result)

            elif self._is_likely_label(result.text):
                # Keep other alphabetic text that might be labels
                labels.append(result)

        return labels

    def _is_likely_label(self, text: str) -> bool:
        """
        Check if text is likely a room label or annotation.

        Args:
            text: Text to check

        Returns:
            True if text appears to be a label
        """
        if not text or len(text) < 2:
            return False

        # Clean text
        clean = text.replace(' ', '').upper()

        # Count alphabetic characters
        alpha_count = sum(1 for c in clean if c.isalpha())
        ratio = alpha_count / len(clean) if clean else 0

        # At least 60% alphabetic and at least 3 chars
        return ratio >= 0.6 and len(clean) >= 3

    def extract_all_categorized(
        self,
        image: Image.Image
    ) -> dict[str, list[OCRResult]]:
        """
        Extract all text and categorize by type.

        Returns a dictionary with:
        - 'dimensions': Numeric text with units
        - 'labels': Room names and labels
        - 'other': Uncategorized text

        Args:
            image: PIL Image to process

        Returns:
            Dictionary mapping category to list of OCRResult
        """
        all_text = self.extract_text_regions(image)

        categorized = {
            'dimensions': [],
            'labels': [],
            'other': []
        }

        for result in all_text:
            # Try as dimension first
            corrected = OCRCorrector.fix_dimension_text(result.text)
            value, unit = OCRCorrector.extract_numeric_value(corrected)

            if value is not None:
                result.text = corrected
                categorized['dimensions'].append(result)
                continue

            # Try as room label
            matched = OCRCorrector.fuzzy_match_room_name(
                result.text,
                STANDARD_ROOM_NAMES
            )

            if matched:
                result.text = matched
                categorized['labels'].append(result)
                continue

            # Check if likely label
            if self._is_likely_label(result.text):
                categorized['labels'].append(result)
                continue

            # Otherwise, categorize as other
            categorized['other'].append(result)

        return categorized


def extract_with_paddle_ocr(
    image: Image.Image,
    lang: str = 'en'
) -> dict[str, list[OCRResult]]:
    """
    Convenience function to extract and categorize text using PaddleOCR.

    Args:
        image: PIL Image to process
        lang: Language code

    Returns:
        Dictionary with 'dimensions', 'labels', and 'other' categories.
        Returns empty dict if PaddleOCR is not available.
    """
    extractor = OCRExtractor(lang=lang)

    if not extractor.available:
        return {'dimensions': [], 'labels': [], 'other': []}

    return extractor.extract_all_categorized(image)


_paddle_ocr_available = None  # Cache the result

def is_paddle_ocr_available() -> bool:
    """
    Check if PaddleOCR is installed and available.

    Returns:
        True if PaddleOCR can be used
    """
    global _paddle_ocr_available

    # Return cached result if available
    if _paddle_ocr_available is not None:
        return _paddle_ocr_available

    try:
        # Just check if the module exists without full import
        import importlib.util
        spec = importlib.util.find_spec("paddleocr")
        _paddle_ocr_available = spec is not None
    except Exception:
        _paddle_ocr_available = False

    return _paddle_ocr_available
