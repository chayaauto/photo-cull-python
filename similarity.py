from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Cosine similarity utilities for L2-normalized embedding vectors."""

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    @staticmethod
    def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
        """Return an (n, n) matrix of cosine similarities."""
        return embeddings @ embeddings.T

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - SimilarityCalculator.cosine_similarity(a, b)


@dataclass(frozen=True)
class GroupSimilarityStats:
    group_index: int
    member_ids: list[str]
    size: int
    avg_similarity: float | None
    min_similarity: float | None
    max_similarity: float | None


def _pairwise_similarities_in_group(embeddings: np.ndarray, indices: list[int]) -> list[float]:
    if len(indices) < 2:
        return []
    sub = embeddings[indices]
    sims: list[float] = []
    n = len(sub)
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(float(sub[i] @ sub[j]))
    return sims


def group_similarity_stats(
    image_ids: list[str],
    embeddings: np.ndarray,
    index_groups: list[list[int]],
) -> list[GroupSimilarityStats]:
    stats: list[GroupSimilarityStats] = []
    for group_index, indices in enumerate(index_groups):
        member_ids = [image_ids[i] for i in indices]
        sims = _pairwise_similarities_in_group(embeddings, indices)
        stats.append(
            GroupSimilarityStats(
                group_index=group_index,
                member_ids=member_ids,
                size=len(indices),
                avg_similarity=sum(sims) / len(sims) if sims else None,
                min_similarity=min(sims) if sims else None,
                max_similarity=max(sims) if sims else None,
            )
        )
    return stats


def overall_avg_intra_group_similarity(group_stats: list[GroupSimilarityStats]) -> float | None:
    """Mean of per-group average similarities (groups with size >= 2 only)."""
    avgs = [g.avg_similarity for g in group_stats if g.avg_similarity is not None]
    if not avgs:
        return None
    return sum(avgs) / len(avgs)


def log_pairwise_similarities(image_ids: list[str], embeddings: np.ndarray) -> None:
    """Debug-only: log cosine similarity for every image pair."""
    n = len(image_ids)
    logger.info("debug_similarity pairwise dump image_count=%d pairs=%d", n, n * (n - 1) // 2)
    for i in range(n):
        for j in range(i + 1, n):
            sim = SimilarityCalculator.cosine_similarity(embeddings[i], embeddings[j])
            logger.info(
                "debug_similarity image_a=%s image_b=%s similarity=%.4f",
                image_ids[i],
                image_ids[j],
                sim,
            )
