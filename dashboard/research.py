"""Approved research data from a local file or read-only deployment mount.

This module never downloads, modifies, uploads, or substitutes research data.
The checksum identifies the approved processed input, not hardcoded findings.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
from typing import Mapping

DEFAULT_FILENAME = "dataset_preprocessed_v3_clean.csv"
APPROVED_SHA256 = "6164d277eece96037fb4d47972d08aed2e7b67c1eef88d8383aeaf0cc3c80ae0"


class ResearchDataError(ValueError):
    """Deployment data must be provisioned or corrected by the project owner."""


@dataclass(frozen=True)
class ResearchSource:
    path: Path
    content: bytes
    sha256: str


def read_research_source(root: Path, environ: Mapping[str, str] | None = None) -> ResearchSource:
    settings = os.environ if environ is None else environ
    configured = settings.get("SENTILOKAL_RESEARCH_DATA_PATH") or settings.get("SHOPEE_DATA_PATH")
    path = Path(configured).expanduser() if configured else root / "data" / DEFAULT_FILENAME
    expected = settings.get("SENTILOKAL_RESEARCH_DATA_SHA256", APPROVED_SHA256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ResearchDataError("The research data checksum configuration is invalid. Please contact the project owner.")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ResearchDataError("The official research dataset is not available on this server. The project owner must complete the dataset setup.") from exc
    digest = sha256(content).hexdigest()
    if digest != expected:
        raise ResearchDataError("The research file does not match the approved cleaned dataset. Please contact the project owner to verify the configured file.")
    return ResearchSource(path=path, content=content, sha256=digest)
