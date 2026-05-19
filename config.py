from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CLUSTERING_DISTANCE_THRESHOLD = 0.22
DEFAULT_CLIP_MODEL_NAME = "clip-ViT-B-32"
DEFAULT_GROUPING_ENGINE = "clip"

MIN_CLUSTERING_DISTANCE_THRESHOLD = 0.01
MAX_CLUSTERING_DISTANCE_THRESHOLD = 1.5


@dataclass(frozen=True)
class AppConfig:
    clustering_distance_threshold: float
    clip_model_name: str
    debug_similarity: bool
    grouping_engine: str

    def log_summary(self) -> None:
        logger.info(
            "Config clustering_distance_threshold=%.4f clip_model_name=%s "
            "grouping_engine=%s debug_similarity=%s",
            self.clustering_distance_threshold,
            self.clip_model_name,
            self.grouping_engine,
            self.debug_similarity,
        )


_config: AppConfig | None = None


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_distance_threshold(raw: str | None) -> float:
    if raw is None or not raw.strip():
        return DEFAULT_CLUSTERING_DISTANCE_THRESHOLD
    try:
        value = float(raw)
    except ValueError as e:
        raise ValueError(
            f"CLUSTERING_DISTANCE_THRESHOLD must be a number, got {raw!r}"
        ) from e
    if not MIN_CLUSTERING_DISTANCE_THRESHOLD <= value <= MAX_CLUSTERING_DISTANCE_THRESHOLD:
        raise ValueError(
            "CLUSTERING_DISTANCE_THRESHOLD must be between "
            f"{MIN_CLUSTERING_DISTANCE_THRESHOLD} and "
            f"{MAX_CLUSTERING_DISTANCE_THRESHOLD} (cosine distance), got {value}"
        )
    return value


def _parse_clip_model_name(raw: str | None) -> str:
    name = (raw or "").strip() or DEFAULT_CLIP_MODEL_NAME
    if len(name) < 3:
        raise ValueError(f"CLIP_MODEL_NAME must be a non-empty model id, got {raw!r}")
    return name


def _parse_grouping_engine(raw: str | None) -> str:
    engine = (raw or DEFAULT_GROUPING_ENGINE).strip().lower()
    if engine not in ("clip", "hash"):
        raise ValueError(
            f"GROUPING_ENGINE must be 'clip' or 'hash', got {engine!r}"
        )
    return engine


def load_config() -> AppConfig:
    return AppConfig(
        clustering_distance_threshold=_parse_distance_threshold(
            os.environ.get("CLUSTERING_DISTANCE_THRESHOLD")
        ),
        clip_model_name=_parse_clip_model_name(os.environ.get("CLIP_MODEL_NAME")),
        debug_similarity=_parse_bool(
            os.environ.get("DEBUG_SIMILARITY", "false")
        ),
        grouping_engine=_parse_grouping_engine(os.environ.get("GROUPING_ENGINE")),
    )


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def validate_config_on_startup() -> AppConfig:
    """Load and validate configuration; fail fast before serving traffic."""
    config = get_config()
    config.log_summary()
    return config
