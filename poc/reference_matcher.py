"""
Reference matching engine for the PoC.

Uses CLIP embeddings + FAISS for similarity search across the
sample reference pool library.  Falls back to structural (OpenCV)
features when CLIP is not installed.
"""

import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ReferencePool:
    """Metadata for one reference pool folder."""
    folder_id: str
    folder_path: str
    line_drawing_path: str
    dxf_path: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class MatchResult:
    """Single match result."""
    reference: ReferencePool
    score: float
    rank: int


class ReferenceMatcher:
    """
    Match an input pool sketch to the nearest reference using embeddings.

    Supports two backends:
      - CLIP (preferred, requires open_clip + torch)
      - Structural fallback (OpenCV contour features, always available)
    """

    def __init__(self, sample_dir: str):
        self.sample_dir = sample_dir
        self.references: list[ReferencePool] = []
        self.index = None  # FAISS index or None
        self.embeddings: Optional[np.ndarray] = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._use_clip = False

        self._load_references()
        self._try_init_clip()
        self._build_index()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_references(self):
        """Scan sample_data folders and load metadata."""
        sample_path = Path(self.sample_dir)
        if not sample_path.exists():
            logger.warning(f"Sample dir not found: {self.sample_dir}")
            return

        for folder in sorted(sample_path.iterdir()):
            if not folder.is_dir():
                continue
            line_png = folder / "line_drawing.png"
            dxf_file = folder / "final.dxf"
            meta_file = folder / "metadata.json"

            if not line_png.exists():
                continue

            meta = {}
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)

            self.references.append(ReferencePool(
                folder_id=folder.name,
                folder_path=str(folder),
                line_drawing_path=str(line_png),
                dxf_path=str(dxf_file) if dxf_file.exists() else "",
                metadata=meta,
            ))

        logger.info(f"Loaded {len(self.references)} reference pools")

    # ------------------------------------------------------------------
    # CLIP backend
    # ------------------------------------------------------------------
    def _try_init_clip(self):
        """Try to load CLIP model. Falls back to structural if unavailable."""
        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._use_clip = True
            logger.info("CLIP model loaded successfully")
        except ImportError:
            logger.info("open_clip not installed — using structural matching fallback")
            self._use_clip = False
        except Exception as e:
            logger.warning(f"CLIP init failed: {e} — using structural fallback")
            self._use_clip = False

    def _embed_clip(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for an image."""
        import torch
        img_tensor = self._clip_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self._clip_model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Structural fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _structural_features(image: Image.Image) -> np.ndarray:
        """
        Extract simple structural features using OpenCV.
        Returns a fixed-length feature vector.
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
            area / (256 * 256),                              # normalized area
            perimeter / (4 * 256),                           # normalized perimeter
            hull_area / (256 * 256) if hull_area > 0 else 0, # hull area
            area / hull_area if hull_area > 0 else 0,        # solidity
            w / h if h > 0 else 1,                           # aspect ratio
            len(approx),                                     # corner count
            4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,  # circularity
            len(contours),                                   # contour count
        ], dtype=np.float32)

        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        return features

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed image using best available backend."""
        if self._use_clip:
            return self._embed_clip(image)
        return self._structural_features(image)

    def _build_index(self):
        """Build FAISS index (or numpy fallback) from reference embeddings."""
        if not self.references:
            return

        embeddings = []
        for ref in self.references:
            try:
                img = Image.open(ref.line_drawing_path).convert("RGB")
                emb = self._embed_image(img)
                ref.embedding = emb
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Failed to embed {ref.folder_id}: {e}")
                embeddings.append(np.zeros_like(embeddings[0]) if embeddings else np.zeros(8))

        self.embeddings = np.array(embeddings, dtype=np.float32)

        # Try FAISS, fall back to numpy cosine
        try:
            import faiss
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            # Normalize for cosine similarity via inner product
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")
        except ImportError:
            logger.info("FAISS not installed — using numpy cosine fallback")
            self.index = None

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def match(self, input_image: Image.Image, top_k: int = 5) -> list[MatchResult]:
        """
        Find the top-K most similar reference pools.

        Args:
            input_image: PIL Image of the input pool sketch
            top_k: Number of results to return

        Returns:
            Ranked list of MatchResult
        """
        if not self.references or self.embeddings is None:
            return []

        query = self._embed_image(input_image).reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self.references))

        if self.index is not None:
            import faiss
            faiss.normalize_L2(query)
            scores, indices = self.index.search(query, k)
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < 0:
                    continue
                results.append(MatchResult(
                    reference=self.references[idx],
                    score=float(score),
                    rank=rank + 1,
                ))
            return results
        else:
            # Numpy fallback: cosine similarity
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            emb_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
            sims = (emb_norm @ query_norm.T).flatten()
            top_idx = np.argsort(-sims)[:k]
            return [
                MatchResult(
                    reference=self.references[idx],
                    score=float(sims[idx]),
                    rank=rank + 1,
                )
                for rank, idx in enumerate(top_idx)
            ]

    @property
    def backend(self) -> str:
        """Return which embedding backend is active."""
        return "CLIP" if self._use_clip else "Structural (OpenCV)"

    @property
    def num_references(self) -> int:
        return len(self.references)
