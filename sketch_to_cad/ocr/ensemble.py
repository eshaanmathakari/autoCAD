"""
Multi-engine OCR with consensus voting for dimension extraction.

Runs PaddleOCR and Gemini Vision, clusters results by spatial
proximity, and votes on the best text within each cluster.
"""

import math
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sketch_to_cad.ocr.imperial_parser import ImperialDimension
from sketch_to_cad.ocr.detection import TextRegion, PaddleDetector
from sketch_to_cad.ocr.recognition import GeminiRecognizer
from sketch_to_cad.ocr.postprocessing import reparse_with_cleaning, filter_dimensions

logger = logging.getLogger(__name__)


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
        return self._norm_x if hasattr(self, "_norm_x") else 0

    @property
    def norm_y(self) -> int:
        return self._norm_y if hasattr(self, "_norm_y") else 0

    def set_norm(self, img_w: int, img_h: int):
        self._norm_x = int(self.center_x / img_w * 1000) if img_w else 0
        self._norm_y = int(self.center_y / img_h * 1000) if img_h else 0


class OCREnsemble:
    """Multi-engine OCR with consensus voting for dimension extraction."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        correction_dict: Optional[dict[str, str]] = None,
    ):
        self._detector = PaddleDetector()
        self._recognizer = GeminiRecognizer(api_key=gemini_api_key)
        self._corrections = correction_dict or {}

    @staticmethod
    def _cluster_detections(
        detections: list[TextRegion],
        distance_threshold: int = 30,
    ) -> list[list[TextRegion]]:
        """Cluster detections by spatial proximity."""
        if not detections:
            return []

        used = [False] * len(detections)
        clusters: list[list[TextRegion]] = []

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
    def _vote(cluster: list[TextRegion]) -> DimensionResult:
        """Vote on the best text within a cluster."""
        texts = Counter()
        for det in cluster:
            key = det.text.strip()
            if det.parsed:
                key = f"{det.parsed.total_inches}"
            texts[key] += 1

        best_key = texts.most_common(1)[0][0]
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

    def extract_dimensions(self, image: Image.Image) -> list[DimensionResult]:
        """
        Run all OCR engines, cluster by position, vote on text.

        Returns dimensions with consensus confidence.
        """
        all_detections: list[TextRegion] = []

        paddle_results = self._detector.detect(image)
        gemini_results = self._recognizer.recognize(image)

        # Clean and re-parse with corrections
        for det in paddle_results + gemini_results:
            cleaned = reparse_with_cleaning(det, self._corrections)
            if cleaned.parsed is not None:
                all_detections.append(cleaned)

        if not all_detections:
            return []

        clusters = self._cluster_detections(all_detections, distance_threshold=40)
        results = [self._vote(cluster) for cluster in clusters]

        w, h = image.size
        for r in results:
            r.set_norm(w, h)

        results.sort(key=lambda r: (r.center_y, r.center_x))
        return results

    @property
    def available_engines(self) -> list[str]:
        engines = []
        if self._detector.available:
            engines.append("PaddleOCR")
        if self._recognizer.available:
            engines.append("Gemini Vision")
        return engines
