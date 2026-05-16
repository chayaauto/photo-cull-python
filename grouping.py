from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ImageFingerprint:
    image_id: str
    phash: imagehash.ImageHash
    width: int
    height: int
    file_size_bytes: int


def phash_from_path(path: Path, image_id: str) -> ImageFingerprint:
    file_size_bytes = path.stat().st_size
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            h_val = imagehash.phash(im)
    except UnidentifiedImageError as e:
        raise ValueError(f"Unrecognized image format for id={image_id}") from e
    return ImageFingerprint(
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
    fingerprints: list[ImageFingerprint],
    threshold: int,
) -> list[list[ImageFingerprint]]:
    n = len(fingerprints)
    if n == 0:
        return []
    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = fingerprints[i].phash - fingerprints[j].phash
            if dist <= threshold:
                _uf_union(parent, i, j)
    roots: dict[int, list[ImageFingerprint]] = {}
    for idx in range(n):
        r = _uf_find(parent, idx)
        roots.setdefault(r, []).append(fingerprints[idx])
    return list(roots.values())


def recommend_id(members: list[ImageFingerprint]) -> str:
    """
    Prefer highest resolution (pixel count), then larger file size, then stable id tie-break.
    """
    best = max(
        members,
        key=lambda m: (m.width * m.height, m.file_size_bytes, m.image_id),
    )
    return best.image_id
