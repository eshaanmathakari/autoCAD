"""
Image processing utilities for sketch preparation.

Handles:
- Loading images from various sources (file upload, path)
- Preprocessing for optimal API consumption
- Image enhancement for better line detection
- Edge detection for isolating structural lines
"""

from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
from typing import BinaryIO

import cv2
import numpy as np

from .config import MAX_IMAGE_DIMENSION


class ImageProcessor:
    """Handle image loading and preprocessing for sketch analysis."""

    @classmethod
    def load_from_upload(cls, uploaded_file: BinaryIO) -> Image.Image:
        """
        Load image from a Streamlit uploaded file or file-like object.

        Args:
            uploaded_file: File-like object (e.g., Streamlit UploadedFile)

        Returns:
            PIL Image object, preprocessed for API consumption
        """
        image = Image.open(uploaded_file)
        return cls._preprocess(image)

    @classmethod
    def load_from_path(cls, file_path: str) -> Image.Image:
        """
        Load image from a file path.

        Args:
            file_path: Path to the image file

        Returns:
            PIL Image object, preprocessed for API consumption
        """
        image = Image.open(file_path)
        return cls._preprocess(image)

    @classmethod
    def load_from_bytes(cls, image_bytes: bytes) -> Image.Image:
        """
        Load image from bytes.

        Args:
            image_bytes: Raw image bytes

        Returns:
            PIL Image object, preprocessed for API consumption
        """
        image = Image.open(BytesIO(image_bytes))
        return cls._preprocess(image)

    @classmethod
    def _preprocess(cls, image: Image.Image) -> Image.Image:
        """
        Preprocess image for optimal API consumption.

        Operations:
        - Convert to RGB if needed (removes alpha channel)
        - Resize if too large (preserve aspect ratio)

        Args:
            image: PIL Image object

        Returns:
            Preprocessed PIL Image
        """
        # Ensure RGB mode (API doesn't handle RGBA/P/L well)
        if image.mode != "RGB":
            # Handle images with transparency by compositing on white
            if image.mode in ("RGBA", "P"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "P":
                    image = image.convert("RGBA")
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert("RGB")

        # Resize if too large (preserve aspect ratio)
        if max(image.size) > MAX_IMAGE_DIMENSION:
            ratio = MAX_IMAGE_DIMENSION / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    @classmethod
    def enhance_for_extraction(cls, image: Image.Image, contrast: float = 1.5) -> Image.Image:
        """
        Apply enhancement filters for better line detection.

        Useful for:
        - Faint pencil sketches
        - Low contrast scans
        - Photos with poor lighting

        Args:
            image: PIL Image to enhance
            contrast: Contrast multiplier (1.0 = no change, >1 = more contrast)

        Returns:
            Enhanced PIL Image
        """
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)

        # Slight sharpening to make lines crisper
        image = image.filter(ImageFilter.SHARPEN)

        return image

    @classmethod
    def to_bytes(cls, image: Image.Image, format: str = "PNG") -> bytes:
        """
        Convert PIL Image to bytes.

        Args:
            image: PIL Image object
            format: Output format (PNG, JPEG, etc.)

        Returns:
            Image as bytes
        """
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return buffer.getvalue()

    @classmethod
    def get_dimensions(cls, image: Image.Image) -> tuple[int, int]:
        """
        Get image dimensions.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (width, height)
        """
        return image.size

    @classmethod
    def extract_lines_only(cls, image: Image.Image) -> Image.Image:
        """
        Extract only the line work from an image using edge detection.

        Removes colored shading, textures, and fills to isolate
        structural lines (walls, boundaries).

        Args:
            image: PIL Image (RGB)

        Returns:
            PIL Image with only lines (black on white)
        """
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply adaptive thresholding to extract lines
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=10
        )

        # Morphological operations to clean up noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Convert back to PIL
        return Image.fromarray(cleaned).convert("RGB")

    @classmethod
    def to_grayscale(cls, image: Image.Image) -> Image.Image:
        """
        Convert image to grayscale.

        Args:
            image: PIL Image

        Returns:
            Grayscale PIL Image (RGB mode for API compatibility)
        """
        gray = image.convert("L")
        return gray.convert("RGB")

    @classmethod
    def detect_edges(cls, image: Image.Image) -> Image.Image:
        """
        Apply Canny edge detection to highlight boundaries.

        Args:
            image: PIL Image

        Returns:
            PIL Image with edges highlighted (white on black)
        """
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Invert so lines are black on white (like original drawing)
        inverted = cv2.bitwise_not(edges)

        return Image.fromarray(inverted).convert("RGB")
