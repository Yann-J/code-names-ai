from __future__ import annotations

import gzip
import logging
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_FASTTEXT_URLS: dict[str, str] = {
    "en": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz",
}


def fasttext_default_filename(language: str) -> str:
    return f"cc.{language}.300.bin"


def download_fasttext(language: str, dest_dir: Path, *, force: bool = False) -> Path:
    """Download the official Facebook fastText `.bin` for `language` into `dest_dir`.

    The remote file is `.bin.gz`; we stream it to disk and decompress in place.
    Returns the path to the decompressed `.bin`.
    """
    url = _FASTTEXT_URLS.get(language)
    if url is None:
        raise ValueError(
            f"No fastText URL configured for language {language!r}. "
            f"Add an entry to embedding.download._FASTTEXT_URLS."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    bin_path = dest_dir / fasttext_default_filename(language)
    gz_path = bin_path.with_suffix(bin_path.suffix + ".gz")

    if bin_path.exists() and not force:
        logger.info("fastText already at %s", bin_path)
        return bin_path

    logger.info("downloading %s → %s", url, gz_path)
    with urllib.request.urlopen(url) as response, gz_path.open("wb") as out:
        shutil.copyfileobj(response, out)

    logger.info("decompressing → %s", bin_path)
    with gzip.open(gz_path, "rb") as src, bin_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink()

    return bin_path
