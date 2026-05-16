import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

from grouping import ImageFingerprint, cluster_by_hash_distance, phash_from_path, recommend_id
from models import GroupImagesRequest, GroupImagesResponse, ImageGroup
from utils import ca_bundle_path, download_image_to_temp, unlink_quiet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image grouping", version="0.1.0")


@app.on_event("startup")
def _log_ssl_bundle() -> None:
    logger.info("SSL CA bundle: %s", ca_bundle_path())


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

    image_ids = [img.id for img in body.images]
    logger.info(
        "group-images start count=%d threshold=%d ids=%s",
        len(body.images),
        body.hash_distance_threshold,
        image_ids,
    )

    paths: list[Path] = []
    try:
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
        fingerprints: list[ImageFingerprint] = []
        for img, path in zip(body.images, paths, strict=True):
            try:
                fp = phash_from_path(path, img.id)
                logger.info(
                    "Hashed id=%s %dx%d bytes=%d phash=%s",
                    img.id,
                    fp.width,
                    fp.height,
                    fp.file_size_bytes,
                    fp.phash,
                )
            except ValueError as e:
                logger.exception("Hash failed for id=%s path=%s", img.id, path)
                raise HTTPException(status_code=422, detail=str(e)) from e
            fingerprints.append(fp)

        clusters = cluster_by_hash_distance(
            fingerprints,
            threshold=body.hash_distance_threshold,
        )
        logger.info("Clustered into %d group(s)", len(clusters))

        groups: list[ImageGroup] = []
        for cluster in clusters:
            ids = sorted(m.image_id for m in cluster)
            rec = recommend_id(cluster)
            logger.info("Group ids=%s recommended=%s", ids, rec)
            groups.append(ImageGroup(image_ids=ids, recommended_id=rec))
        groups.sort(key=lambda g: (g.image_ids[0] if g.image_ids else "", g.recommended_id))

        logger.info("group-images done groups=%d", len(groups))
        return GroupImagesResponse(groups=groups)
    finally:
        for p in paths:
            unlink_quiet(p)
