"""
OCR ensemble for the PoC.

Two-engine setup:
  1. PaddleOCR (if installed) — general text recognition
  2. Gemini Vision — targeted dimension extraction via prompt

Results are clustered by position and consensus-voted.
"""

import os
import sys
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

# Add parent src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc.imperial_parser import parse as parse_imperial, ImperialDimension

logger = logging.getLogger(__name__)


@dataclass
class OCRDetection:
    """Single OCR detection from one engine."""
    text: str
    center_x: int       # pixel coords
    center_y: int
    confidence: float
    engine: str          # "paddle" or "gemini"
    parsed: Optional[ImperialDimension] = None


@dataclass
class DimensionResult:
    """Consensus dimension result after voting."""
    text: str
    parsed: Optional[ImperialDimension]
    center_x: int
    center_y: int
    confidence: float
    needs_review: bool
    engines_agreed: int
    total_engines: int

    @property
    def norm_x(self) -> int:
        """Placeholder — call set_norm with image dims."""
        return self._norm_x if hasattr(self, "_norm_x") else 0

    @property
    def norm_y(self) -> int:
        return self._norm_y if hasattr(self, "_norm_y") else 0

    def set_norm(self, img_w: int, img_h: int):
        self._norm_x = int(self.center_x / img_w * 1000) if img_w else 0
        self._norm_y = int(self.center_y / img_h * 1000) if img_h else 0


class OCREnsemble:
    """
    Multi-engine OCR with consensus voting for dimension extraction.
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        self._paddle = None
        self._gemini_key = gemini_api_key or os.environ.get("GOOGLE_API_KEY")
        self._init_paddle()

    def _init_paddle(self):
        try:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
            )
            logger.info("PaddleOCR initialized")
        except Exception as e:
            logger.info(f"PaddleOCR not available: {e}")

    # ------------------------------------------------------------------
    # Engine 1: PaddleOCR
    # ------------------------------------------------------------------
    def _run_paddle(self, image: Image.Image) -> list[OCRDetection]:
        if self._paddle is None:
            return []
        try:
            img_array = np.array(image.convert("RGB"))
            results = self._paddle.ocr(img_array, cls=True)
            detections = []
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, conf) = line[0], line[1]
                    cx = int(sum(p[0] for p in bbox) / 4)
                    cy = int(sum(p[1] for p in bbox) / 4)
                    parsed = parse_imperial(text)
                    detections.append(OCRDetection(
                        text=text,
                        center_x=cx,
                        center_y=cy,
                        confidence=float(conf),
                        engine="paddle",
                        parsed=parsed,
                    ))
            return detections
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Engine 2: Gemini Vision (targeted prompt)
    # ------------------------------------------------------------------
    def _run_gemini(self, image: Image.Image) -> list[OCRDetection]:
        if not self._gemini_key:
            return []
        try:
            from google import genai
            from google.genai import types
            import io
            import json

            client = genai.Client(api_key=self._gemini_key)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            w, h = image.size

            prompt = f"""Analyze this swimming pool sketch image ({w}x{h} pixels).
Extract ALL visible dimension annotations / measurements.
Pool dimensions use feet and inches notation like: 2', 2'4", 12'6", 4", etc.

Return a JSON array where each element is:
{{"text": "<dimension text as written>", "x": <center x pixel>, "y": <center y pixel>}}

Only return dimension/measurement text. Ignore labels like "POOL" or "STAIRS".
Return ONLY the JSON array, no other text."""

            response = client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
                contents=[
                    types.Content(parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(
                            mime_type="image/png", data=image_bytes
                        )),
                    ])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            raw = response.text.strip()
            items = json.loads(raw)
            detections = []
            for item in items:
                text = str(item.get("text", ""))
                x = int(item.get("x", 0))
                y = int(item.get("y", 0))
                parsed = parse_imperial(text)
                detections.append(OCRDetection(
                    text=text,
                    center_x=x,
                    center_y=y,
                    confidence=0.85,  # Gemini doesn't return per-item confidence
                    engine="gemini",
                    parsed=parsed,
                ))
            return detections
        except Exception as e:
            logger.warning(f"Gemini OCR failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Clustering & consensus
    # ------------------------------------------------------------------
    @staticmethod
    def _cluster_detections(
        detections: list[OCRDetection],
        distance_threshold: int = 30,
    ) -> list[list[OCRDetection]]:
        """Cluster detections by spatial proximity."""
        if not detections:
            return []

        used = [False] * len(detections)
        clusters: list[list[OCRDetection]] = []

        for i, det in enumerate(detections):
            if used[i]:
                continue
            cluster = [det]
            used[i] = True
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                dist = math.hypot(
                    det.center_x - detections[j].center_x,
                    det.center_y - detections[j].center_y,
                )
                if dist < distance_threshold:
                    cluster.append(detections[j])
                    used[j] = True
            clusters.append(cluster)

        return clusters

    @staticmethod
    def _vote(cluster: list[OCRDetection]) -> DimensionResult:
        """Vote on the best text within a cluster."""
        # Group by parsed total_inches (if parseable) or raw text
        from collections import Counter
        texts = Counter()
        for det in cluster:
            key = det.text.strip()
            if det.parsed:
                key = f"{det.parsed.total_inches}"
            texts[key] += 1

        best_key = texts.most_common(1)[0][0]
        # Find the detection matching the best key with highest confidence
        best_det = None
        for det in cluster:
            key = det.text.strip()
            if det.parsed:
                key = f"{det.parsed.total_inches}"
            if key == best_key:
                if best_det is None or det.confidence > best_det.confidence:
                    best_det = det

        engines = set(d.engine for d in cluster)
        agreed = texts.most_common(1)[0][1]
        avg_x = int(np.mean([d.center_x for d in cluster]))
        avg_y = int(np.mean([d.center_y for d in cluster]))
        avg_conf = float(np.mean([d.confidence for d in cluster]))

        # Boost confidence when multiple engines agree
        if len(engines) > 1 and agreed >= 2:
            avg_conf = min(avg_conf + 0.1, 1.0)

        return DimensionResult(
            text=best_det.text if best_det else cluster[0].text,
            parsed=best_det.parsed if best_det else cluster[0].parsed,
            center_x=avg_x,
            center_y=avg_y,
            confidence=avg_conf,
            needs_review=avg_conf < 0.9,
            engines_agreed=agreed,
            total_engines=len(engines),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_dimensions(self, image: Image.Image) -> list[DimensionResult]:
        """
        Run all OCR engines, cluster by position, vote on text.

        Returns dimensions with consensus confidence.
        """
        all_detections: list[OCRDetection] = []

        # Run engines
        paddle_results = self._run_paddle(image)
        gemini_results = self._run_gemini(image)

        # Keep only dimension-like detections (those that parse as imperial)
        for det in paddle_results + gemini_results:
            if det.parsed is not None:
                all_detections.append(det)

        if not all_detections:
            return []

        # Cluster and vote
        clusters = self._cluster_detections(all_detections, distance_threshold=40)
        results = [self._vote(cluster) for cluster in clusters]

        # Set normalized coordinates
        w, h = image.size
        for r in results:
            r.set_norm(w, h)

        # Sort by position (top-to-bottom, left-to-right)
        results.sort(key=lambda r: (r.center_y, r.center_x))

        return results

    @property
    def available_engines(self) -> list[str]:
        engines = []
        if self._paddle is not None:
            engines.append("PaddleOCR")
        if self._gemini_key:
            engines.append("Gemini Vision")
        return engines
