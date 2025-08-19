# core/speciesnet_api.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Literal
import threading
from speciesnet import SpeciesNet, DEFAULT_MODEL
from speciesnet.utils import prepare_instances_dict

_DEFAULT_COMPONENTS: Literal["classifier","detector","ensemble","all"] = "all"
_DEFAULT_RUN_MODE: Literal["multi_thread","multi_process"] = "multi_thread"

class _SpeciesNetSingleton:
    _instance = None
    _lock = threading.Lock()
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.model = SpeciesNet(
                    DEFAULT_MODEL,
                    components=_DEFAULT_COMPONENTS,
                    geofence=True,
                    multiprocessing=False,  # Streamlit-safe
                )
            return cls._instance

_singleton = _SpeciesNetSingleton()

def _canon_str(p: str | Path) -> str:
    return str(Path(p).expanduser().resolve())

def _match_key(preds: Dict, target_abs: str):
    """Return the prediction entry matching target_abs; fall back to first value."""
    if not preds:
        return {}
    if target_abs in preds:
        return preds[target_abs]
    # try canonicalizing keys
    for k, v in preds.items():
        try:
            if _canon_str(k) == target_abs:
                return v
        except Exception:
            pass
    # last resort: first value
    return next(iter(preds.values()))

def run_speciesnet_one(
    image_path: str | Path,
    *,
    country: Optional[str] = None,        # prefer ISO-3166 alpha-3 (e.g., "USA")
    admin1_region: Optional[str] = None,  # e.g., "CO"
    components: Literal["classifier","detector","ensemble","all"] = _DEFAULT_COMPONENTS,
) -> Dict:
    p_abs = _canon_str(image_path)

    # Quick normalization: if someone passes "US", drop to None rather than wrong code
    if country and len(country) == 2:
        country = None

    instances = prepare_instances_dict(
        instances_json=None,
        filepaths=[p_abs],
        filepaths_txt=None,
        folders=None,
        folders_txt=None,
        country=country,
        admin1_region=admin1_region,
    )

    model = _singleton.model if components == _DEFAULT_COMPONENTS else SpeciesNet(
        DEFAULT_MODEL, components=components, geofence=True, multiprocessing=False
    )

    # Try the normal combined pipeline first (detector + classifier)
    preds = model.predict(
        instances_dict=instances,
        run_mode=_DEFAULT_RUN_MODE,
        batch_size=8,
        progress_bars=False,
        predictions_json=None,   # in-memory only
    )

    if preds:
        return _match_key(preds, p_abs)

    # Fallback: force detect -> classify -> ensemble_from_past_runs
    detections = model.detect(
        instances_dict=instances,
        run_mode=_DEFAULT_RUN_MODE,
        progress_bars=False,
        predictions_json=None,
    ) or {}

    classifications = model.classify(
        instances_dict=instances,
        detections_dict=detections,
        run_mode=_DEFAULT_RUN_MODE,
        batch_size=8,
        progress_bars=False,
        predictions_json=None,
    ) or {}

    ensemble = model.ensemble_from_past_runs(
        instances_dict=instances,
        classifications_dict=classifications,
        detections_dict=detections,
        progress_bars=False,
        predictions_json=None,
    ) or {}

    return _match_key(ensemble or classifications or detections, p_abs)
