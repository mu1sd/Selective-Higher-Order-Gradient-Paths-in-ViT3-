from __future__ import annotations

import hashlib
import json
from pathlib import Path


PATH_MODES = ("GSHPS_AUTO", "GSHPS_SENS", "GSHPS_50", "RAND_PATH")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_mask(config):
    path = Path(config["path_mask_path"]).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != config["path_mask_sha256"]:
        raise RuntimeError("path mask hash differs from frozen config")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("path mask has no entries")
    for index, entry in enumerate(entries):
        if int(entry.get("block_index", -1)) != index:
            raise RuntimeError("path mask block indexes are not contiguous")
        if not isinstance(entry.get("swiglu"), bool) or not isinstance(entry.get("dwc"), bool):
            raise RuntimeError("path mask values must be booleans")
    return payload, {"path": str(path), "sha256": actual, "mask_hash": canonical_hash(entries)}


def apply_classification_mask(model, entries):
    from models.gradient_modes import set_model_gradient_path_mask
    return set_model_gradient_path_mask(model, entries)


def apply_segmentation_mask(model, entries):
    return model.backbone.set_gradient_path_mask(entries)
