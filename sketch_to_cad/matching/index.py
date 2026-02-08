"""
Reference matching engine.

Uses CLIP embeddings + FAISS for similarity search across the
reference pool library. Falls back to structural (OpenCV)
features when CLIP is not installed.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from sketch_to_cad.matching.embeddings import CLIPEmbedder, StructuralEmbedder

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
        self.index = None
        self.embeddings: Optional[np.ndarray] = None

        self._clip = CLIPEmbedder()
        self._structural = StructuralEmbedder()
        self._use_clip = self._clip.available

        self._load_references()
        self._build_index()

    def _load_references(self):
        """Scan template folders and load metadata."""
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

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed image using best available backend."""
        if self._use_clip:
            return self._clip.embed(image)
        return self._structural.embed(image)

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

        try:
            import faiss
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")
        except ImportError:
            logger.info("FAISS not installed — using numpy cosine fallback")
            self.index = None

    def match(self, input_image: Image.Image, top_k: int = 5) -> list[MatchResult]:
        """Find the top-K most similar reference pools."""
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
        return "CLIP" if self._use_clip else "Structural (OpenCV)"

    @property
    def num_references(self) -> int:
        return len(self.references)
