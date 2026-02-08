"""
Image preprocessing for reference matching.

Contour normalization bridges the domain gap between
hand-drawn sketches and clean template line drawings.
"""

import numpy as np
from PIL import Image


def normalize_contours(image: Image.Image, target_size: int = 256) -> np.ndarray:
    """
    Binarize, extract contours, redraw on clean canvas with uniform stroke.

    This collapses both hand-drawn and template images into similar-looking
    contour images, reducing the domain gap before embedding.
    """
    import cv2

    img = np.array(image.convert("L"))
    img = cv2.resize(img, (target_size, target_size))
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, 255, 2)

    return canvas
