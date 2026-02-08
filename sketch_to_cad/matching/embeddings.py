"""
Image embedding backends for reference matching.

CLIPEmbedder: Uses OpenCLIP ViT-B/32 (preferred, requires torch).
StructuralEmbedder: OpenCV contour features (always available fallback).
"""

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """CLIP/OpenCLIP embedding backend."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self._model = None
        self._preprocess = None
        self._available = False
        try:
            import open_clip
            import torch  # noqa: F401

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._available = True
            logger.info("CLIP model loaded successfully")
        except ImportError:
            logger.info("open_clip not installed — CLIPEmbedder unavailable")
        except Exception as e:
            logger.warning(f"CLIP init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def dimension(self) -> int:
        return 512

    def embed(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for an image."""
        import torch

        img_tensor = self._preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self._model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()


class StructuralEmbedder:
    """OpenCV structural features — always-available fallback."""

    @property
    def available(self) -> bool:
        return True

    @property
    def dimension(self) -> int:
        return 8

    def embed(self, image: Image.Image) -> np.ndarray:
        """
        Extract simple structural features using OpenCV.
        Returns a fixed-length 8D feature vector.
        """
        import cv2

        img = np.array(image.convert("L"))
        img = cv2.resize(img, (256, 256))
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(8, dtype=np.float32)

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(largest)
        approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)

        features = np.array([
            area / (256 * 256),
            perimeter / (4 * 256),
            hull_area / (256 * 256) if hull_area > 0 else 0,
            area / hull_area if hull_area > 0 else 0,
            w / h if h > 0 else 1,
            len(approx),
            4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,
            len(contours),
        ], dtype=np.float32)

        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        return features
