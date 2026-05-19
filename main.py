import asyncio
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import validate_config_on_startup
from grouping import (
    ClipGroupingEngine,
    GroupingEngine,
    HashGroupingEngine,
    default_grouping_engine,
    recommend_id,
)
from models import GroupImagesRequest, GroupImagesResponse, ImageGroup
from utils import ca_bundle_path, download_image_to_temp, unlink_quiet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_IMAGES_PER_REQUEST = 500

app = FastAPI(title="Image grouping", version="0.2.0")

_cors_origins = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_grouping_engine: GroupingEngine | None = None


def grouping_engine() -> GroupingEngine:
    global _grouping_engine
    if _grouping_engine is None:
        _grouping_engine = default_grouping_engine()
    return _grouping_engine


def engine_for_request(body: GroupImagesRequest) -> GroupingEngine:
    from config import get_config

    if get_config().grouping_engine == "hash":
        return HashGroupingEngine(threshold=body.hash_distance_threshold)
    return grouping_engine()


@app.on_event("startup")
def _startup() -> None:
    validate_config_on_startup()
    logger.info("SSL CA bundle: %s", ca_bundle_path())
    engine = grouping_engine()
    logger.info("Grouping engine ready: %s", type(engine).__name__)
    if type(engine).__name__ == "ClipGroupingEngine":
        from embeddings import EmbeddingGenerator

        EmbeddingGenerator.instance()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _download_one(image_id: str, url: str) -> Path:
    try:
        return await download_image_to_temp(url, image_id=image_id)
    except ValueError:
        logger.exception("Download rejected for id=%s url=%s", image_id, url)
        raise
    except RuntimeError:
        logger.exception("Download failed for id=%s url=%s", image_id, url)
        raise


@app.post("/group-images", response_model=GroupImagesResponse)
async def group_images(body: GroupImagesRequest) -> GroupImagesResponse:
    if not body.images:
        raise HTTPException(status_code=400, detail="images must not be empty")

    if len(body.images) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_IMAGES_PER_REQUEST} images per request",
        )

    image_ids = [img.id for img in body.images]
    request_started = time.perf_counter()
    logger.info(
        "group-images start image_count=%d ids=%s",
        len(body.images),
        image_ids,
    )

    paths: list[Path] = []
    try:
        download_started = time.perf_counter()
        tasks = [_download_one(img.id, str(img.url)) for img in body.images]
        try:
            downloaded = await asyncio.gather(*tasks)
        except RuntimeError as e:
            detail = f"Failed to download an image: {e}"
            logger.error("group-images download error: %s", detail)
            raise HTTPException(status_code=502, detail=detail) from e
        except ValueError as e:
            logger.error("group-images size limit: %s", e)
            raise HTTPException(status_code=413, detail=str(e)) from e

        paths = list(downloaded)
        download_sec = time.perf_counter() - download_started
        logger.info(
            "group-images downloads complete image_count=%d download_seconds=%.3f",
            len(paths),
            download_sec,
        )

        group_started = time.perf_counter()
        engine = engine_for_request(body)
        try:
            clusters = await asyncio.to_thread(engine.group, paths, image_ids)
        except ValueError as e:
            logger.exception("Grouping failed")
            raise HTTPException(status_code=422, detail=str(e)) from e
        group_sec = time.perf_counter() - group_started

        embedding_sec: float | None = None
        clustering_sec: float | None = None
        if isinstance(engine, ClipGroupingEngine):
            embedding_sec = engine.last_embedding_seconds
            clustering_sec = engine.last_clustering_seconds
            logger.info(
                "group-images embeddings_seconds=%.3f clustering_seconds=%.3f "
                "groups_generated=%d",
                embedding_sec or 0.0,
                clustering_sec or 0.0,
                len(clusters),
            )
        else:
            logger.info(
                "group-images grouping_seconds=%.3f groups_generated=%d",
                group_sec,
                len(clusters),
            )

        groups: list[ImageGroup] = []
        for cluster in clusters:
            ids = sorted(m.image_id for m in cluster)
            rec = recommend_id(cluster)
            logger.info("Group ids=%s recommended=%s", ids, rec)
            groups.append(ImageGroup(image_ids=ids, recommended_id=rec))
        groups.sort(key=lambda g: (g.image_ids[0] if g.image_ids else "", g.recommended_id))

        total_sec = time.perf_counter() - request_started
        logger.info(
            "group-images done image_count=%d groups_generated=%d "
            "embedding_seconds=%s clustering_seconds=%s total_seconds=%.3f",
            len(body.images),
            len(groups),
            f"{embedding_sec:.3f}" if embedding_sec is not None else "n/a",
            f"{clustering_sec:.3f}" if clustering_sec is not None else "n/a",
            total_sec,
        )
        return GroupImagesResponse(groups=groups)
    finally:
        for p in paths:
            unlink_quiet(p)
