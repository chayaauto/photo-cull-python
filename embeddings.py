from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer

from config import get_config
from grouping import ImageRecord

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


class EmbeddingGenerator:
    """Loads a CLIP model once and produces L2-normalized embedding vectors."""

    _instance: EmbeddingGenerator | None = None

    def __init__(
        self,
        model_name: str | None = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        name = model_name or get_config().clip_model_name
        self._batch_size = batch_size
        logger.info("Loading CLIP model model=%s", name)
        self._model = SentenceTransformer(name)
        logger.info("CLIP model ready model=%s", name)

    @classmethod
    def instance(cls) -> EmbeddingGenerator:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_pil_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return shape (n, dim) float32 embeddings, L2-normalized per row."""
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            images,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def build_records(self, paths: list[Path], image_ids: list[str]) -> list[ImageRecord]:
        """Load images from disk, attach metadata and normalized embeddings."""
        if len(paths) != len(image_ids):
            raise ValueError("paths and image_ids length mismatch")

        pil_images: list[Image.Image] = []
        meta: list[tuple[int, int, int]] = []

        for path, image_id in zip(paths, image_ids, strict=True):
            file_size_bytes = path.stat().st_size
            try:
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                    w, h = rgb.size
                    pil_images.append(rgb.copy())
            except UnidentifiedImageError as e:
                raise ValueError(f"Unrecognized image format for id={image_id}") from e
            meta.append((w, h, file_size_bytes))

        embeddings = self.embed_pil_images(pil_images)
        for img in pil_images:
            img.close()

        records: list[ImageRecord] = []
        for i, image_id in enumerate(image_ids):
            w, h, size = meta[i]
            records.append(
                ImageRecord(
                    image_id=image_id,
                    width=w,
                    height=h,
                    file_size_bytes=size,
                    embedding=embeddings[i],
                )
            )
        return records
