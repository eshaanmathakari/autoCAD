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

from .config import (
    GOOGLE_API_KEY, GEMINI_MODEL, EXTRACTION_PROMPT,
    EXTRACTION_PROMPT_WALLS, EXTRACTION_PROMPT_TEXT
)
from .geometry_models import SketchGeometry, SketchMetadata
from .image_processor import ImageProcessor


class ExtractionError(Exception):
    """Custom exception for geometry extraction failures."""
    pass


class GeminiGeometryExtractor:
    """
    Extract geometric primitives from sketch images using Gemini Vision.

    Uses Gemini's multimodal capabilities to analyze hand-drawn sketches
    and return structured geometry data.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """
        Initialize the Gemini extractor.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Gemini model ID (defaults to GEMINI_MODEL env var)

        Raises:
            ValueError: If no API key is provided
        """
        self.api_key = api_key or GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model or GEMINI_MODEL
        self.client = genai.Client(api_key=self.api_key)

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

    def extract(self, image: Image.Image, prompt: str | None = None) -> str:
        """
        Extract geometry from a sketch image.

        Args:
            image: PIL Image of the hand-drawn sketch
            prompt: Custom extraction prompt (defaults to EXTRACTION_PROMPT)

        Returns:
            JSON string conforming to SketchGeometry schema

        Raises:
            ExtractionError: If API call fails
        """
        extraction_prompt = prompt or EXTRACTION_PROMPT

        # Get image dimensions for metadata
        width, height = ImageProcessor.get_dimensions(image)

        # Convert image to bytes for API
        image_bytes = ImageProcessor.to_bytes(image, format="PNG")

        try:
            # Call Gemini with structured output
            response = self.client.models.generate_content(
                model=self.model,
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
                )
            )

            return response.text

        except Exception as e:
            raise ExtractionError(f"Gemini API call failed: {str(e)}") from e

    def extract_validated(
        self,
        image: Image.Image,
        prompt: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> SketchGeometry:
        """
        Extract geometry with validation and retry logic.

        Args:
            image: PIL Image of the sketch
            prompt: Custom extraction prompt
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Validated SketchGeometry object

        Raises:
            ExtractionError: If extraction fails after all retries
        """
        last_error: Exception | None = None
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

    def extract_raw(self, image: Image.Image, prompt: str | None = None) -> dict:
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
        preprocessed_image: Image.Image | None = None,
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

        # Pass 1: Extract walls from preprocessed image
        walls_geometry = None
        for attempt in range(max_retries):
            try:
                walls_json = self.extract(preprocessed_image, EXTRACTION_PROMPT_WALLS)
                walls_geometry = SketchGeometry.model_validate_json(walls_json)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    walls_geometry = SketchGeometry(
                        metadata=SketchMetadata(confidence_score=0.0),
                        entities=[],
                        warnings=[f"Wall extraction failed: {e}"]
                    )

        # Pass 2: Extract text from original image
        text_geometry = None
        for attempt in range(max_retries):
            try:
                text_json = self.extract(original_image, EXTRACTION_PROMPT_TEXT)
                text_geometry = SketchGeometry.model_validate_json(text_json)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    text_geometry = SketchGeometry(
                        metadata=SketchMetadata(confidence_score=0.0),
                        entities=[],
                        warnings=[f"Text extraction failed: {e}"]
                    )

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
