"""Utilities for applying one frozen gradient mode to every official TTT block."""

from .ttt_block import CANONICAL_GRADIENT_MODES, normalize_gradient_mode


def set_model_gradient_mode(model, mode):
    """Apply ``mode`` to all compatible TTT modules and return their count."""
    canonical = normalize_gradient_mode(mode)
    count = 0
    for module in model.modules():
        setter = getattr(module, "set_gradient_mode", None)
        if setter is not None:
            setter(canonical)
            count += 1
    if count == 0:
        raise RuntimeError("model contains no module exposing set_gradient_mode()")
    return count


def gradient_mode_registry():
    """Stable machine-readable registry used by preflight and audit code."""
    return dict(CANONICAL_GRADIENT_MODES)


def set_model_gradient_path_mask(model, entries):
    """Apply ordered block×branch entries to official TTT modules."""
    modules = [
        module for module in model.modules()
        if getattr(module, "set_high_order_path_mask", None) is not None
    ]
    if len(modules) != len(entries):
        raise RuntimeError(f"path-mask block mismatch: model={len(modules)} mask={len(entries)}")
    for index, (module, entry) in enumerate(zip(modules, entries)):
        if int(entry.get("block_index", index)) != index:
            raise RuntimeError(f"non-canonical block index at {index}")
        module.set_high_order_path_mask(entry["swiglu"], entry["dwc"])
    return len(modules)
