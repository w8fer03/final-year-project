"""Refresh the application's chart code after updates in a running Streamlit process.

Streamlit can rerun app.py while Python retains an older dashboard.charts in
sys.modules. Reading our fixed local source also avoids timestamp-based bytecode
caches when a Git checkout replaces code without changing its size or timestamp.
"""
from __future__ import annotations

from hashlib import sha256
from importlib import import_module
from pathlib import Path
from threading import RLock
from types import ModuleType


_CHARTS_PATH = Path(__file__).resolve().with_name("charts.py")
_REQUIRED_CHARTS = (
    "sentiment_donut", "rating_bars", "rating_imbalance", "review_length_chart",
    "trend_chart", "product_comparison", "confusion_heatmap", "metric_comparison",
)
_LOCK = RLock()


def _read_source() -> bytes:
    return _CHARTS_PATH.read_bytes()


def refresh_charts() -> ModuleType:
    """Return current local charts, repairing stale imports without reloading each rerun.

    This intentionally accepts no module name, path, or uploaded content. Only the
    repository's own charts.py can be executed. The existing module and dictionary
    remain in place so references held by the app and other imports stay current.
    """
    with _LOCK:
        charts = import_module("dashboard.charts")
        if Path(charts.__file__).resolve() != _CHARTS_PATH:
            raise ImportError("The dashboard chart module is not the local application module.")
        source = _read_source()
        fingerprint = sha256(source).hexdigest()
        missing = any(not callable(getattr(charts, name, None)) for name in _REQUIRED_CHARTS)
        if getattr(charts, "_sentilokal_source_sha256", None) != fingerprint or missing:
            # Compile before modifying the live module. Execute exact source bytes,
            # rather than accepting a potentially stale timestamp-based .pyc file.
            code = compile(source, str(_CHARTS_PATH), "exec")
            exec(code, charts.__dict__)
            if any(not callable(getattr(charts, name, None)) for name in _REQUIRED_CHARTS):
                raise ImportError("The dashboard chart source is incomplete. Update all project files together.")
            charts._sentilokal_source_sha256 = fingerprint
        return charts
