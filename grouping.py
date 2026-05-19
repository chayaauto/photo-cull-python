from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    width: int
    height: int
    file_size_bytes: int
    embedding: np.ndarray | None = None
    phash: imagehash.ImageHash | None = None


# Backward-compatible alias
ImageFingerprint = ImageRecord


def phash_from_path(path: Path, image_id: str) -> ImageRecord:
    file_size_bytes = path.stat().st_size
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            h_val = imagehash.phash(im)
    except UnidentifiedImageError as e:
        raise ValueError(f"Unrecognized image format for id={image_id}") from e
    return ImageRecord(
        image_id=image_id,
        phash=h_val,
        width=w,
        height=h,
        file_size_bytes=file_size_bytes,
    )


def _uf_find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _uf_union(parent: list[int], i: int, j: int) -> None:
    ri, rj = _uf_find(parent, i), _uf_find(parent, j)
    if ri != rj:
        parent[rj] = ri


def cluster_by_hash_distance(
    fingerprints: list[ImageRecord],
    threshold: int,
) -> list[list[ImageRecord]]:
    n = len(fingerprints)
    if n == 0:
        return []
    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = fingerprints[i].phash, fingerprints[j].phash
            if pi is None or pj is None:
                continue
            if pi - pj <= threshold:
                _uf_union(parent, i, j)
    roots: dict[int, list[ImageRecord]] = {}
    for idx in range(n):
        r = _uf_find(parent, idx)
        roots.setdefault(r, []).append(fingerprints[idx])
    return list(roots.values())


def recommend_id(members: list[ImageRecord]) -> str:
    """Highest resolution, then file size, then stable id ordering."""
    best = max(
        members,
        key=lambda m: (m.width * m.height, m.file_size_bytes, m.image_id),
    )
    return best.image_id


class GroupingEngine(ABC):
    @abstractmethod
    def group(
        self,
        paths: list[Path],
        image_ids: list[str],
    ) -> list[list[ImageRecord]]:
        ...


class HashGroupingEngine(GroupingEngine):
    def __init__(self, threshold: int = 10) -> None:
        self.threshold = threshold

    def group(
        self,
        paths: list[Path],
        image_ids: list[str],
    ) -> list[list[ImageRecord]]:
        records = [phash_from_path(p, iid) for p, iid in zip(paths, image_ids, strict=True)]
        return cluster_by_hash_distance(records, threshold=self.threshold)


class ClipGroupingEngine(GroupingEngine):
    def __init__(
        self,
        embedding_generator: object | None = None,
        clustering_engine: object | None = None,
    ) -> None:
        from clustering import ClusteringEngine
        from embeddings import EmbeddingGenerator

        self._embedder = embedding_generator or EmbeddingGenerator.instance()
        self._clusterer = clustering_engine or ClusteringEngine()
        self.last_embedding_seconds: float | None = None
        self.last_clustering_seconds: float | None = None

    def group(
        self,
        paths: list[Path],
        image_ids: list[str],
    ) -> list[list[ImageRecord]]:
        embed_started = time.perf_counter()
        records = self._embedder.build_records(paths, image_ids)
        self.last_embedding_seconds = time.perf_counter() - embed_started

        cluster_started = time.perf_counter()
        embeddings = np.stack([r.embedding for r in records], axis=0)
        index_groups = self._clusterer.cluster_indices(
            embeddings,
            image_ids=image_ids,
        )
        self.last_clustering_seconds = time.perf_counter() - cluster_started

        logger.info(
            "Clip grouping embedding_seconds=%.3f clustering_seconds=%.3f",
            self.last_embedding_seconds,
            self.last_clustering_seconds,
        )
        return [[records[i] for i in group] for group in index_groups]


def default_grouping_engine() -> GroupingEngine:
    from config import get_config

    engine_name = get_config().grouping_engine
    if engine_name == "hash":
        threshold = int(os.environ.get("HASH_DISTANCE_THRESHOLD", "10"))
        logger.info("Using HashGroupingEngine threshold=%d", threshold)
        return HashGroupingEngine(threshold=threshold)
    logger.info("Using ClipGroupingEngine")
    return ClipGroupingEngine()
