"""
Gemini Vision API integration for geometry extraction.

Uses Google's Gemini 3 model to analyze sketch images and extract
geometric primitives in structured JSON format.
"""

import base64
import json
import time
from io import BytesIO
from typing import Any

from PIL import Image
from google import genai
from google.genai import types
from pydantic import ValidationError

from typing import Optional, Tuple

from .config import (
    GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_FALLBACK_MODEL, GEMINI_REQUEST_TIMEOUT_SEC,
    EXTRACTION_PROMPT, EXTRACTION_PROMPT_WALLS, EXTRACTION_PROMPT_TEXT
)
from .geometry_models import (
    SketchGeometry, SketchMetadata, TextEntity, DimensionEntity,
    Point2D, LayerType
)
from .image_processor import ImageProcessor
from .validation import GeometryValidator, ValidationResult
from .calibration import ScaleCalibrator, CalibrationResult

# Optional extractor abstraction (GeometryExtractor protocol)
GeometryExtractorProtocol = Optional[Any]  # Optional[GeometryExtractor] from extractors.base


class ExtractionError(Exception):
    """Custom exception for geometry extraction failures."""
    pass


class GeminiGeometryExtractor:
    """
    Extract geometric primitives from sketch images using Gemini Vision.

    Uses Gemini's multimodal capabilities to analyze hand-drawn sketches
    and return structured geometry data.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        geometry_extractor: GeometryExtractorProtocol = None,
    ):
        """
        Initialize the Gemini extractor.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Gemini model ID (defaults to GEMINI_MODEL env var)
            geometry_extractor: Optional extractor (e.g. CompositeExtractor).
                               When set, extract_hybrid uses it instead of extract_two_pass.

        Raises:
            ValueError: If no API key is provided and geometry_extractor is not set
        """
        self.api_key = api_key or GOOGLE_API_KEY
        self.geometry_extractor = geometry_extractor
        if not self.api_key and not geometry_extractor:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter, or provide a geometry_extractor."
            )
        self.model = model or GEMINI_MODEL
        self.fallback_model = GEMINI_FALLBACK_MODEL
        self.request_timeout_sec = GEMINI_REQUEST_TIMEOUT_SEC
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_schema(self) -> dict[str, Any]:
        """
        Build JSON schema for Gemini structured output.

        Returns the Pydantic model's JSON schema.
        """
        return SketchGeometry.model_json_schema()

    def extract(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Extract geometry from a sketch image.

        Args:
            image: PIL Image of the hand-drawn sketch
            prompt: Custom extraction prompt (defaults to EXTRACTION_PROMPT)
            model_override: Optional model ID to use instead of self.model (e.g. fallback)

        Returns:
            JSON string conforming to SketchGeometry schema

        Raises:
            ExtractionError: If API call fails
        """
        if self.client is None:
            raise ExtractionError(
                "No API key configured. Set GOOGLE_API_KEY or use a geometry_extractor."
            )
        extraction_prompt = prompt or EXTRACTION_PROMPT
        model = model_override or self.model

        # Get image dimensions for metadata
        width, height = ImageProcessor.get_dimensions(image)

        # Convert image to bytes for API
        image_bytes = ImageProcessor.to_bytes(image, format="PNG")

        try:
            # Call Gemini with structured output and explicit timeout (avoids 60s default disconnect)
            response = self.client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=extraction_prompt),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=image_bytes
                                )
                            )
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=self._build_schema(),
                    temperature=0.1,  # Low temperature for precision
                ),
                http_options=types.HttpOptions(timeout=self.request_timeout_sec),
            )

            return response.text

        except Exception as e:
            raise ExtractionError(f"Gemini API call failed: {str(e)}") from e

    def extract_validated(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        apply_validation: bool = True,
        apply_snap: bool = True
    ) -> SketchGeometry:
        """
        Extract geometry with validation and retry logic.

        Args:
            image: PIL Image of the sketch
            prompt: Custom extraction prompt
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            apply_validation: Whether to apply OCR corrections and validation
            apply_snap: Whether to snap dimensions to standard values

        Returns:
            Validated SketchGeometry object

        Raises:
            ExtractionError: If extraction fails after all retries
        """
        last_error: Optional[Exception] = None
        width, height = ImageProcessor.get_dimensions(image)

        for attempt in range(max_retries):
            try:
                # Get JSON from API
                json_result = self.extract(image, prompt)

                # Parse and validate
                geometry = SketchGeometry.model_validate_json(json_result)

                # Add image dimensions to metadata if not present
                if geometry.metadata.image_width is None:
                    geometry.metadata.image_width = width
                if geometry.metadata.image_height is None:
                    geometry.metadata.image_height = height

                # Apply validation and OCR corrections
                if apply_validation:
                    validator = GeometryValidator(width, height)
                    geometry, validation_result = validator.validate_all(
                        geometry, apply_snap=apply_snap
                    )

                    # Add validation warnings to geometry
                    for warning in validation_result.warnings:
                        if warning not in geometry.warnings:
                            geometry.warnings.append(warning)

                    # Adjust confidence based on validation
                    geometry.metadata.confidence_score *= validation_result.confidence_adjustment

                    # Store pixel ratio for calibration if detected
                    if validation_result.pixel_to_unit_ratio:
                        geometry.metadata.drawing_scale = (
                            f"calibrated:{validation_result.pixel_to_unit_ratio:.4f}"
                        )

                return geometry

            except ValidationError as e:
                last_error = ExtractionError(f"Invalid geometry format: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

            except ExtractionError as e:
                last_error = e
                # Check for rate limiting
                if "rate" in str(e).lower() or "quota" in str(e).lower():
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                elif attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    break

            except Exception as e:
                last_error = ExtractionError(f"Unexpected error: {str(e)}")
                break

        raise ExtractionError(
            f"Extraction failed after {max_retries} attempts. Last error: {last_error}"
        )

    def extract_raw(self, image: Image.Image, prompt: Optional[str] = None) -> dict:
        """
        Extract geometry and return as raw dictionary (no validation).

        Useful for debugging or when you want to handle validation yourself.

        Args:
            image: PIL Image of the sketch
            prompt: Custom extraction prompt

        Returns:
            Dictionary parsed from JSON response
        """
        json_result = self.extract(image, prompt)
        return json.loads(json_result)

    def extract_two_pass(
        self,
        original_image: Image.Image,
        preprocessed_image: Optional[Image.Image] = None,
        max_retries: int = 2
    ) -> SketchGeometry:
        """
        Two-pass extraction for noisy hand-drawn plans.

        Pass 1: Extract walls/geometry from preprocessed (line-only) image
        Pass 2: Extract text/labels from original image

        Args:
            original_image: Original color image
            preprocessed_image: Line-extracted image (if None, auto-generate)
            max_retries: Retries per pass

        Returns:
            Merged SketchGeometry with both geometry and text
        """
        from .geometry_models import GeometryEntity

        width, height = ImageProcessor.get_dimensions(original_image)

        # Generate preprocessed image if not provided
        if preprocessed_image is None:
            preprocessed_image = ImageProcessor.extract_lines_only(original_image)

        # Retry with exponential backoff on disconnect; optionally try fallback model on last failure
        def _run_pass(image: Image.Image, prompt: str, pass_name: str) -> Optional[SketchGeometry]:
            last_err: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    json_str = self.extract(image, prompt, model_override=None)
                    return SketchGeometry.model_validate_json(json_str)
                except Exception as e:
                    last_err = e
                    is_disconnect = "disconnected" in str(e).lower() or "remoteprotocolerror" in type(e).__name__.lower()
                    if is_disconnect and attempt < max_retries - 1:
                        backoff = 2.0 * (2 ** attempt)
                        time.sleep(backoff)
                        continue
                    break
            # Optional: try once with faster fallback model
            try:
                json_str = self.extract(image, prompt, model_override=self.fallback_model)
                return SketchGeometry.model_validate_json(json_str)
            except Exception:
                pass
            return SketchGeometry(
                metadata=SketchMetadata(confidence_score=0.0),
                entities=[],
                warnings=[f"{pass_name} extraction failed: {last_err}"]
            )

        # Pass 1: Extract walls from preprocessed image
        walls_geometry = _run_pass(preprocessed_image, EXTRACTION_PROMPT_WALLS, "Wall")

        # Pass 2: Extract text from original image
        text_geometry = _run_pass(original_image, EXTRACTION_PROMPT_TEXT, "Text")

        # Local line fallback when both passes failed (no geometry from Gemini)
        if not walls_geometry.entities and walls_geometry.warnings:
            from .local_line_extractor import extract_lines_from_image
            local_geom = extract_lines_from_image(
                preprocessed_image,
                width,
                height,
            )
            if local_geom.entities:
                walls_geometry = local_geom

        # Merge results
        merged_entities: list[GeometryEntity] = []
        merged_warnings: list[str] = []

        if walls_geometry:
            merged_entities.extend(walls_geometry.entities)
            merged_warnings.extend(walls_geometry.warnings)

        if text_geometry:
            merged_entities.extend(text_geometry.entities)
            merged_warnings.extend(text_geometry.warnings)

        # Calculate merged confidence
        wall_conf = walls_geometry.metadata.confidence_score if walls_geometry else 0.0
        text_conf = text_geometry.metadata.confidence_score if text_geometry else 0.0
        merged_confidence = (wall_conf + text_conf) / 2 if (wall_conf + text_conf) > 0 else 0.0

        # Get title from text pass if available
        title = None
        if text_geometry and text_geometry.metadata.title:
            title = text_geometry.metadata.title

        return SketchGeometry(
            metadata=SketchMetadata(
                title=title,
                detected_units="mm",
                sketch_type="2d_orthographic",
                image_width=width,
                image_height=height,
                confidence_score=merged_confidence
            ),
            entities=merged_entities,
            warnings=merged_warnings
        )

    def extract_hybrid(
        self,
        original_image: Image.Image,
        preprocessed_image: Optional[Image.Image] = None,
        use_paddle_ocr: bool = True,
        apply_validation: bool = True,
        apply_snap: bool = False,
        apply_calibration: bool = True
    ) -> Tuple[SketchGeometry, Optional[CalibrationResult]]:
        """
        Hybrid extraction combining Gemini vision with PaddleOCR.

        Flow:
        1. Use Gemini for geometry (walls, shapes) via two-pass extraction
        2. Use PaddleOCR for text/dimension extraction (if available)
        3. Calibrate scale from RAW dimension values (BEFORE any corrections)
        4. Validate and apply corrections (optional snapping, default OFF)

        Args:
            original_image: Original color image
            preprocessed_image: Line-extracted image (optional, auto-generated if None)
            use_paddle_ocr: Whether to use PaddleOCR for text extraction
            apply_validation: Whether to apply OCR corrections and validation
            apply_snap: Whether to snap dimensions to standard values (default False - can corrupt values)
            apply_calibration: Whether to compute scale calibration

        Returns:
            Tuple of (validated SketchGeometry, CalibrationResult or None)
        """
        width, height = ImageProcessor.get_dimensions(original_image)

        # Step 1: Extract geometry (composite extractor or Gemini two-pass)
        if self.geometry_extractor is not None:
            geometry = self.geometry_extractor.extract(original_image, preprocessed_image)
        else:
            geometry = self.extract_two_pass(original_image, preprocessed_image)

        # Step 2: Enhance text extraction with PaddleOCR
        if use_paddle_ocr:
            geometry = self._enhance_with_paddle_ocr(
                geometry, original_image, width, height
            )

        # Step 3: Calibrate scale from RAW dimensions BEFORE any validation/correction
        # This ensures scale is calculated from original extracted values
        calibration = None
        if apply_calibration:
            calibrator = ScaleCalibrator(original_image)
            # Get dimensions BEFORE any modification
            dimensions = [e for e in geometry.entities if e.type == "dimension"]

            calibration = calibrator.calibrate(
                metadata_scale=geometry.metadata.drawing_scale,
                dimension_entities=dimensions
            )

            if calibration:
                geometry.metadata.drawing_scale = (
                    f"{calibration.method}:1:{int(calibration.scale_factor)}"
                )
                geometry.warnings.append(
                    f"Scale calibrated: 1:{int(calibration.scale_factor)} "
                    f"({calibration.reference_description})"
                )

        # Step 4: Apply validation and OCR corrections AFTER calibration
        if apply_validation:
            validator = GeometryValidator(width, height)
            geometry, validation_result = validator.validate_all(
                geometry, apply_snap=apply_snap
            )

            # Add validation info to geometry
            for warning in validation_result.warnings:
                if warning not in geometry.warnings:
                    geometry.warnings.append(warning)

            geometry.metadata.confidence_score *= validation_result.confidence_adjustment

            # Add correction summary
            if validation_result.corrections:
                correction_count = len(validation_result.corrections)
                geometry.warnings.append(
                    f"Applied {correction_count} OCR/validation corrections"
                )

        return geometry, calibration

    def _enhance_with_paddle_ocr(
        self,
        geometry: SketchGeometry,
        image: Image.Image,
        width: int,
        height: int
    ) -> SketchGeometry:
        """
        Enhance geometry with text detections from PaddleOCR.

        Merges PaddleOCR results with Gemini results, avoiding duplicates.

        Args:
            geometry: Existing geometry from Gemini
            image: Original image
            width: Image width
            height: Image height

        Returns:
            Enhanced SketchGeometry
        """
        try:
            from .paddle_ocr import OCRExtractor, is_paddle_ocr_available

            if not is_paddle_ocr_available():
                geometry.warnings.append("PaddleOCR not available, using Gemini-only extraction")
                return geometry

            extractor = OCRExtractor(lang='en')
            if not extractor.available:
                return geometry

            # Extract text with PaddleOCR
            categorized = extractor.extract_all_categorized(image)

            # Add dimensions that don't already exist
            # Use tighter threshold (30px) for position and check text similarity
            for ocr_result in categorized['dimensions']:
                norm_x = int(ocr_result.center_x / width * 1000)
                norm_y = int(ocr_result.center_y / height * 1000)

                # Check both position (30px) and text similarity (100px radius)
                if not self._dimension_exists_near(geometry, norm_x, norm_y, ocr_result.text, pos_threshold=30, text_threshold=100):
                    # Add as dimension entity
                    geometry.entities.append(DimensionEntity(
                        start=Point2D(x=max(0, norm_x - 50), y=norm_y),
                        end=Point2D(x=min(1000, norm_x + 50), y=norm_y),
                        text_position=Point2D(x=norm_x, y=norm_y),
                        value=ocr_result.text,
                        unit="mm",
                        layer=LayerType.DIMENSIONS,
                        confidence=ocr_result.confidence
                    ))

            # Add room labels that don't already exist
            # Use tighter threshold for labels
            for ocr_result in categorized['labels']:
                norm_x = int(ocr_result.center_x / width * 1000)
                norm_y = int(ocr_result.center_y / height * 1000)

                if not self._text_exists_near(geometry, norm_x, norm_y, ocr_result.text, threshold=30):
                    geometry.entities.append(TextEntity(
                        position=Point2D(x=norm_x, y=norm_y),
                        content=ocr_result.text,
                        height=20,
                        rotation=ocr_result.rotation if abs(ocr_result.rotation) > 5 else 0,
                        layer=LayerType.TEXT,
                        confidence=ocr_result.confidence
                    ))

            if categorized['dimensions'] or categorized['labels']:
                geometry.warnings.append(
                    f"PaddleOCR added: {len(categorized['dimensions'])} dimensions, "
                    f"{len(categorized['labels'])} labels"
                )

        except Exception as e:
            geometry.warnings.append(f"PaddleOCR enhancement failed: {e}")

        return geometry

    def _entity_exists_near(
        self,
        geometry: SketchGeometry,
        x: int,
        y: int,
        threshold: int = 30
    ) -> bool:
        """Check if any dimension or text entity exists near the given position."""
        for entity in geometry.entities:
            if entity.type == "dimension":
                if hasattr(entity, 'text_position') and entity.text_position:
                    dist = ((entity.text_position.x - x) ** 2 +
                            (entity.text_position.y - y) ** 2) ** 0.5
                    if dist < threshold:
                        return True
            elif entity.type == "text":
                if hasattr(entity, 'position'):
                    dist = ((entity.position.x - x) ** 2 +
                            (entity.position.y - y) ** 2) ** 0.5
                    if dist < threshold:
                        return True
        return False

    def _dimension_exists_near(
        self,
        geometry: SketchGeometry,
        x: int,
        y: int,
        text: str,
        pos_threshold: int = 30,
        text_threshold: int = 100
    ) -> bool:
        """
        Check if a similar dimension exists near the given position.

        Uses two checks:
        1. Position-only check with tight threshold (30px)
        2. Same text within wider radius (100px) - catches duplicates with slight offset
        """
        text_normalized = text.strip().upper().replace(' ', '')

        for entity in geometry.entities:
            if entity.type == "dimension":
                if hasattr(entity, 'text_position') and entity.text_position:
                    dist = ((entity.text_position.x - x) ** 2 +
                            (entity.text_position.y - y) ** 2) ** 0.5

                    # Check 1: Very close position (likely same detection)
                    if dist < pos_threshold:
                        return True

                    # Check 2: Same text within wider radius (duplicate detection)
                    if dist < text_threshold and hasattr(entity, 'value') and entity.value:
                        existing_normalized = entity.value.strip().upper().replace(' ', '')
                        if existing_normalized == text_normalized:
                            return True

        return False

    def _text_exists_near(
        self,
        geometry: SketchGeometry,
        x: int,
        y: int,
        text: str,
        threshold: int = 50
    ) -> bool:
        """Check if similar text exists near the given position."""
        text_upper = text.upper().strip()

        for entity in geometry.entities:
            if entity.type == "text" and hasattr(entity, 'position'):
                dist = ((entity.position.x - x) ** 2 +
                        (entity.position.y - y) ** 2) ** 0.5
                if dist < threshold:
                    # Also check if content is similar
                    if hasattr(entity, 'content'):
                        existing_upper = entity.content.upper().strip()
                        if existing_upper == text_upper:
                            return True
                        # Fuzzy match
                        if len(existing_upper) > 0 and len(text_upper) > 0:
                            common = sum(1 for a, b in zip(existing_upper, text_upper) if a == b)
                            ratio = common / max(len(existing_upper), len(text_upper))
                            if ratio > 0.7:
                                return True
        return False


def create_mock_geometry(width: int = 1000, height: int = 1000) -> SketchGeometry:
    """
    Create mock geometry for testing without API calls.

    Args:
        width: Mock image width
        height: Mock image height

    Returns:
        SketchGeometry with sample entities
    """
    from .geometry_models import (
        LineEntity, CircleEntity, RectangleEntity, TextEntity,
        Point2D, LayerType
    )

    return SketchGeometry(
        metadata=SketchMetadata(
            title="Mock Sketch",
            detected_units="mm",
            sketch_type="2d_orthographic",
            image_width=width,
            image_height=height,
            confidence_score=0.95
        ),
        entities=[
            LineEntity(
                start=Point2D(x=100, y=100),
                end=Point2D(x=900, y=100),
                layer=LayerType.GEOMETRY,
                confidence=0.98
            ),
            LineEntity(
                start=Point2D(x=900, y=100),
                end=Point2D(x=900, y=700),
                layer=LayerType.GEOMETRY,
                confidence=0.97
            ),
            LineEntity(
                start=Point2D(x=900, y=700),
                end=Point2D(x=100, y=700),
                layer=LayerType.GEOMETRY,
                confidence=0.96
            ),
            LineEntity(
                start=Point2D(x=100, y=700),
                end=Point2D(x=100, y=100),
                layer=LayerType.GEOMETRY,
                confidence=0.95
            ),
            CircleEntity(
                center=Point2D(x=500, y=400),
                radius=100,
                layer=LayerType.GEOMETRY,
                confidence=0.90
            ),
            TextEntity(
                position=Point2D(x=500, y=50),
                content="Sample Part",
                height=30,
                layer=LayerType.TEXT,
                confidence=0.85
            ),
        ],
        warnings=["This is mock data for testing"]
    )
