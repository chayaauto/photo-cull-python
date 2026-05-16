import asyncio
import logging
import os
import ssl
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import certifi

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SEC = 30
MAX_BYTES = 25 * 1024 * 1024  # 25 MiB per image
_USER_AGENT = "photo-cull/0.1"

_NETFREE_CA_BUNDLE = Path.home() / "Documents" / "netfree" / "cacert-bundle-curl-plus-netfree.pem"


def ca_bundle_path() -> str:
    """
    Resolve CA bundle for HTTPS. Priority:
    1. IMAGE_GROUP_CA_BUNDLE
    2. ~/Documents/netfree/cacert-bundle-curl-plus-netfree.pem (if exists)
    3. SSL_CERT_FILE / SSL_CA_BUNDLE / REQUESTS_CA_BUNDLE (if file exists)
    4. certifi
    """
    candidates: list[str] = []
    if path := os.environ.get("IMAGE_GROUP_CA_BUNDLE"):
        candidates.append(path)
    if _NETFREE_CA_BUNDLE.is_file():
        candidates.append(str(_NETFREE_CA_BUNDLE))
    for key in ("SSL_CERT_FILE", "SSL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        if path := os.environ.get(key):
            candidates.append(path)
    candidates.append(certifi.where())

    for path in candidates:
        p = Path(path).expanduser()
        if p.is_file():
            return str(p.resolve())

    raise FileNotFoundError("No CA bundle found (set IMAGE_GROUP_CA_BUNDLE)")


_CA_BUNDLE = ca_bundle_path()


def ssl_context() -> ssl.SSLContext:
    # create_default_context(cafile=...) is broken on some OpenSSL 3.x + NetFree setups;
    # load_verify_locations works (same as curl --cacert).
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(cafile=_CA_BUNDLE)
    return ctx


def _download_sync(url: str, dest: Path) -> int:
    ctx = ssl_context()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=DEFAULT_TIMEOUT_SEC) as resp:
            status = getattr(resp, "status", 200)
            if status >= 400:
                raise urllib.error.HTTPError(
                    url, status, resp.reason, resp.headers, None
                )
            total = 0
            with dest.open("wb") as out:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise ValueError(f"Image exceeds max size ({MAX_BYTES} bytes)")
                    out.write(chunk)
            return total
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(str(e.reason)) from e


async def download_image_to_temp(
    url: str,
    *,
    image_id: str | None = None,
) -> Path:
    """Download URL to a temp file. Caller must delete the path when done."""
    label = f"id={image_id}" if image_id else "image"
    logger.info("Downloading %s url=%s (CA: %s)", label, url, _CA_BUNDLE)

    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
        suffix = ".bin"

    fd, path_str = tempfile.mkstemp(prefix="img_", suffix=suffix)
    path = Path(path_str)
    os.close(fd)

    try:
        total = await asyncio.to_thread(_download_sync, url, path)
    except Exception:
        path.unlink(missing_ok=True)
        logger.exception("Download failed for %s url=%s", label, url)
        raise

    logger.info("Downloaded %s bytes=%d path=%s", label, total, path)
    return path


def unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
