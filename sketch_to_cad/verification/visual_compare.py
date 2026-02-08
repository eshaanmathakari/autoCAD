"""
Visual comparison for layout verification.

Renders a generated DXF to an image, then compares against
the reference sketch using SSIM and Hu moments.
"""

import io
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sketch_to_cad.domain.constants import SSIM_PASS_THRESHOLD, HU_MOMENT_TOLERANCE


@dataclass
class VisualComparisonResult:
    """Result of visual comparison between reference and generated."""
    ssim_score: float
    hu_moment_distance: float
    passed: bool
    overlay_image_bytes: Optional[bytes] = None


def _render_dxf_to_image(dxf_bytes: bytes, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Render DXF to a grayscale numpy array using ezdxf matplotlib backend."""
    import ezdxf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        f.write(dxf_bytes)
        tmp_path = f.name

    try:
        doc = ezdxf.readfile(tmp_path)
        msp = doc.modelspace()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_aspect("equal")
        ax.set_facecolor("white")

        for entity in msp:
            if entity.dxftype() == "LINE":
                s = entity.dxf.start
                e = entity.dxf.end
                ax.plot([s.x, e.x], [s.y, e.y], "k-", linewidth=1)
            elif entity.dxftype() == "LWPOLYLINE":
                pts = list(entity.get_points(format="xy"))
                if entity.closed and pts:
                    pts.append(pts[0])
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "k-", linewidth=1)

        ax.axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert("L").resize(size)
        return np.array(img)
    finally:
        os.unlink(tmp_path)


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two grayscale images."""
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


def _create_overlay(ref_img: np.ndarray, gen_img: np.ndarray) -> bytes:
    """Create a red/green overlay image showing differences."""
    h, w = ref_img.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = ref_img     # red = reference
    overlay[:, :, 1] = gen_img     # green = generated
    overlay[:, :, 2] = 0

    img = Image.fromarray(overlay)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compare_visual(
    reference_image: Image.Image,
    generated_dxf_bytes: bytes,
    ssim_threshold: float = SSIM_PASS_THRESHOLD,
) -> VisualComparisonResult:
    """
    Render DXF to image, compare against reference using SSIM + Hu moments.

    Args:
        reference_image: PIL Image of the reference (sketch or line drawing)
        generated_dxf_bytes: DXF file as bytes
        ssim_threshold: Minimum SSIM score to pass (default 0.85)

    Returns:
        VisualComparisonResult with scores and overlay image
    """
    import cv2

    size = (512, 512)

    # Prepare reference
    ref_gray = np.array(reference_image.convert("L").resize(size))

    # Render generated DXF
    gen_gray = _render_dxf_to_image(generated_dxf_bytes, size=size)

    # SSIM
    ssim = _compute_ssim(ref_gray, gen_gray)

    # Hu moments comparison
    _, ref_binary = cv2.threshold(ref_gray, 128, 255, cv2.THRESH_BINARY_INV)
    _, gen_binary = cv2.threshold(gen_gray, 128, 255, cv2.THRESH_BINARY_INV)

    ref_contours, _ = cv2.findContours(ref_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gen_contours, _ = cv2.findContours(gen_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hu_distance = 1.0
    if ref_contours and gen_contours:
        ref_largest = max(ref_contours, key=cv2.contourArea)
        gen_largest = max(gen_contours, key=cv2.contourArea)
        hu_distance = cv2.matchShapes(ref_largest, gen_largest, cv2.CONTOURS_MATCH_I1, 0)

    passed = ssim >= ssim_threshold and hu_distance <= HU_MOMENT_TOLERANCE

    overlay_bytes = _create_overlay(ref_gray, gen_gray)

    return VisualComparisonResult(
        ssim_score=round(ssim, 4),
        hu_moment_distance=round(hu_distance, 4),
        passed=passed,
        overlay_image_bytes=overlay_bytes,
    )
