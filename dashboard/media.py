"""Local, verified product photos for the landing page.

No assets are downloaded or hotlinked. Missing, invalid or unrelated catalog
entries resolve to an honest empty image state rather than a substitute photo.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
import warnings

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility.
    import tomli as tomllib

from PIL import Image, UnidentifiedImageError

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "products"
CATALOG_PATH = ASSETS_DIR / "catalog.toml"

# Names were supplied by the owner. Categories use only facts in those names.
PRODUCT_MEDIA = {
    "Baseus H17 Encok 3.5mm Wired Earphone": ("baseus-h17", "Wired earphones"),
    "Baseus C17 Type-C Earphones": ("baseus-c17", "Type-C earphones"),
    "Baseus Encok CZ11 / HZ11": ("baseus-cz11-hz11", "Earphones"),
    "Baseus WM01 TWS Wireless Earphones": ("baseus-wm01", "TWS wireless earphones"),
    "Baseus H19": ("baseus-h19", "Earphones"),
}
IMAGE_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


@dataclass(frozen=True)
class ProductMedia:
    """Presentation-only metadata; source listing identifiers are not exposed."""

    path: Path | None
    alt: str
    source_url: str | None
    category: str
    slug: str
    warnings: tuple[str, ...] = ()
    mime_type: str | None = None

    @property
    def image_data_uri(self) -> str | None:
        """Encode an already validated local image for an escaped HTML card."""
        if self.path is None or self.mime_type is None:
            return None
        try:
            encoded = base64.b64encode(self.path.read_bytes()).decode("ascii")
        except OSError:
            return None
        return f"data:{self.mime_type};base64,{encoded}"


def _source_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if any(c.isspace() or ord(c) < 32 for c in value):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            return None
        if parsed.port not in (None, 443):
            return None
    except ValueError:
        return None
    return value


def product_media(name: str) -> ProductMedia:
    """Resolve one known product's photo, returning a safe empty state on failure.

    Catalog filenames must remain beneath assets/products after resolving
    symlinks. Only actual JPEG, PNG and WebP images are accepted. Validation
    warnings are intended for owner diagnostics rather than product cards.
    """
    known = PRODUCT_MEDIA.get(name)
    if known is None:
        return ProductMedia(None, "Earphones product photo", None, "Earphones", "unknown-product")
    slug, category = known
    alt = f"{name} product photo"

    def empty(message: str | None = None) -> ProductMedia:
        return ProductMedia(None, alt, None, category, slug, (message,) if message else ())

    try:
        # loads handles both tomllib and the Python 3.10 toml fallback.
        catalog = tomllib.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return empty("The local product photo catalog has not been added.")
    except (OSError, ValueError):
        return empty("The local product photo catalog could not be read.")
    entries = catalog.get("products")
    if not isinstance(entries, dict):
        return empty("The product photo catalog requires a products section.")
    entry = entries.get(slug)
    if not isinstance(entry, dict) or entry.get("name") != name:
        return empty("The product photo catalog entry does not match this product.")
    image_path = entry.get("image_path", "")
    if not isinstance(image_path, str):
        return empty("The catalog image_path must be a local filename.")
    if not image_path.strip():
        return empty()
    try:
        candidate = (ASSETS_DIR / image_path).resolve()
        root = ASSETS_DIR.resolve()
        candidate.relative_to(root)
    except (ValueError, OSError, RuntimeError):
        return empty("The catalog image_path must remain inside assets/products.")
    if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        return empty("Product photos must be local JPEG, PNG or WebP images.")
    if not candidate.is_file():
        return empty("The catalog photo file is missing.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(candidate) as image:
                image_format = image.format
                if image_format not in IMAGE_TYPES:
                    return empty("The product photo format is not supported.")
                image.verify()
            with Image.open(candidate) as image:
                image.load()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return empty("The product photo could not be decoded as a valid image.")
    source = _source_url(entry.get("source_url"))
    diagnostics = ()
    if entry.get("source_url") and source is None:
        diagnostics = ("The product photo source must be a valid public HTTPS link.",)
    return ProductMedia(candidate, alt, source, category, slug, diagnostics, IMAGE_TYPES[image_format])
