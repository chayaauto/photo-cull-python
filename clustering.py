from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import get_config
from similarity import (
    group_similarity_stats,
    log_pairwise_similarities,
    overall_avg_intra_group_similarity,
)

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """Groups embedding indices with agglomerative clustering on cosine distance."""

    def __init__(self, distance_threshold: float | None = None) -> None:
        self.distance_threshold = (
            distance_threshold
            if distance_threshold is not None
            else get_config().clustering_distance_threshold
        )

    def cluster_indices(
        self,
        embeddings: np.ndarray,
        *,
        image_ids: list[str] | None = None,
    ) -> list[list[int]]:
        n = embeddings.shape[0]
        if n == 0:
            return []
        if n == 1:
            if image_ids:
                self._log_clustering_summary(
                    image_ids=image_ids,
                    embeddings=embeddings,
                    index_groups=[[0]],
                )
            return [[0]]

        config = get_config()
        if config.debug_similarity and image_ids is not None:
            log_pairwise_similarities(image_ids, embeddings)

        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(embeddings)

        groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            groups.setdefault(int(label), []).append(idx)
        index_groups = list(groups.values())

        if image_ids is not None:
            self._log_clustering_summary(
                image_ids=image_ids,
                embeddings=embeddings,
                index_groups=index_groups,
            )
        else:
            logger.info(
                "clustering complete image_count=%d groups_generated=%d "
                "clustering_distance_threshold=%.4f",
                n,
                len(index_groups),
                self.distance_threshold,
            )

        return index_groups

    def _log_clustering_summary(
        self,
        *,
        image_ids: list[str],
        embeddings: np.ndarray,
        index_groups: list[list[int]],
    ) -> None:
        n = len(image_ids)
        group_sizes = [len(g) for g in index_groups]
        stats = group_similarity_stats(image_ids, embeddings, index_groups)
        overall_avg = overall_avg_intra_group_similarity(stats)

        logger.info(
            "clustering summary image_count=%d groups_generated=%d "
            "group_sizes=%s clustering_distance_threshold=%.4f "
            "overall_avg_intra_group_similarity=%s",
            n,
            len(index_groups),
            group_sizes,
            self.distance_threshold,
            f"{overall_avg:.4f}" if overall_avg is not None else "n/a",
        )

        for g in stats:
            if g.size < 2:
                logger.info(
                    "clustering group_index=%d size=%d member_ids=%s "
                    "avg_intra_group_similarity=n/a (singleton)",
                    g.group_index,
                    g.size,
                    g.member_ids,
                )
                continue
            logger.info(
                "clustering group_index=%d size=%d member_ids=%s "
                "avg_intra_group_similarity=%.4f min_pair_similarity=%.4f "
                "max_pair_similarity=%.4f",
                g.group_index,
                g.size,
                g.member_ids,
                g.avg_similarity,
                g.min_similarity,
                g.max_similarity,
            )
