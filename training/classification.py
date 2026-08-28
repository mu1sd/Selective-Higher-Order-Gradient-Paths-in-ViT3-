#!/usr/bin/env python3
"""Audited single-GPU ImageNet-100 fine-tuning for official H-ViT3-T."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import signal
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


MODES = ("FFFF", "DDDD", "SSSS", "OOOO", "RAND", "GSHPS_AUTO", "GSHPS_SENS", "GSHPS_50", "RAND_PATH")
STOP_REQUESTED = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty index: {path}")
    required = {"relative_path", "label"}
    if not required.issubset(rows[0]):
        raise ValueError(f"index {path} lacks {sorted(required)}")
    return rows


class IndexedImageDataset(Dataset):
    def __init__(self, root: Path, rows, transform, seed: int):
        self.root = root
        self.rows = rows
        self.transform = transform
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["relative_path"]).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(f"index escapes dataset root: {path}") from exc
        local_seed = int(
            hashlib.sha256(
                f"{self.seed}:{self.epoch}:{row['relative_path']}".encode("utf-8")
            ).hexdigest()[:16],
            16,
        ) % (2**31)
        python_state = random.getstate()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(local_seed)
            random.seed(local_seed)
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        random.setstate(python_state)
        return tensor, int(row["label"]), row["relative_path"]


class FrozenBatchSampler(Sampler):
    def __init__(self, rows, seed: int, epoch: int, batch_size: int, start_batch=0):
        ranked = sorted(
            range(len(rows)),
            key=lambda index: hashlib.sha256(
                f"{seed}:{epoch}:{rows[index]['relative_path']}".encode("utf-8")
            ).hexdigest(),
        )
        self.batches = [
            ranked[offset : offset + batch_size]
            for offset in range(0, len(ranked), batch_size)
            if len(ranked[offset : offset + batch_size]) == batch_size
        ][start_batch:]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def transforms_for(size: int):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(int(size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train, evaluation


def load_official_model(config, device):
    official_root = Path(config["official_root"]).resolve()
    sys.path.insert(0, str(official_root / "vittt"))
    from models.gradient_modes import set_model_gradient_mode
    from models.h_vittt import h_vittt_tiny

    seed_all(int(config["seed"]))
    model = h_vittt_tiny(
        drop_path_rate=float(config.get("drop_path_rate", 0.2)),
        use_checkpoint=bool(config.get("use_checkpoint", False)),
    )
    checkpoint_path = Path(config["pretrained_checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    pretrained = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(pretrained, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"official classification checkpoint mismatch: missing={incompatible.missing_keys[:20]} "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    head_seed = int(config.get("head_seed", config["seed"]))
    torch.manual_seed(head_seed)
    model.head = nn.Linear(model.head.in_features, 100)
    from gshps.masks import PATH_MODES, apply_classification_mask, load_mask
    initial_mode = config.get("runtime_initial_mode", "DDDD") if config["mode"] == "RAND" else ("PATH" if config["mode"] in PATH_MODES else config["mode"])
    block_count = set_model_gradient_mode(model, initial_mode)
    if block_count <= 0:
        raise RuntimeError("no official TTT blocks received the gradient mode")
    mask_audit = None
    if config["mode"] in PATH_MODES:
        mask_payload, mask_audit = load_mask(config)
        block_count = apply_classification_mask(model, mask_payload["entries"])
    return model.to(device), {
        "official_root": str(official_root),
        "official_commit": config["official_commit"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "head_seed": head_seed,
        "ttt_block_count": block_count,
        "use_checkpoint": bool(config.get("use_checkpoint", False)),
        "runtime_initial_mode": initial_mode,
        "path_mask": mask_audit,
    }


def set_runtime_mode(model, mode: str) -> int:
    from models.gradient_modes import set_model_gradient_mode

    count = set_model_gradient_mode(model, mode)
    if count <= 0:
        raise RuntimeError(f"no official TTT blocks accepted runtime mode {mode}")
    return count


def load_rand_schedule(config):
    if config["mode"] != "RAND":
        return None, None
    path = Path(config["rand_schedule_path"]).resolve()
    actual_sha = sha256_file(path)
    if actual_sha != config["rand_schedule_sha256"]:
        raise RuntimeError("RAND schedule hash differs from frozen config")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schedule = payload.get("modes") if isinstance(payload, dict) else payload
    if not isinstance(schedule, list) or len(schedule) != int(config["planned_updates"]):
        raise RuntimeError("RAND schedule length differs from planned_updates")
    if any(mode not in ("DDDD", "SSSS") for mode in schedule):
        raise RuntimeError("RAND schedule contains a mode other than DDDD/SSSS")
    counts = {mode: schedule.count(mode) for mode in ("DDDD", "SSSS")}
    if abs(counts["DDDD"] - counts["SSSS"]) > 1:
        raise RuntimeError(f"RAND schedule is not frozen 50/50: {counts}")
    return schedule, {"path": str(path), "sha256": actual_sha, "planned_counts": counts}


def parameter_signature(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                "names": [names[id(parameter)] for parameter in group["params"]],
                "weight_decay": group.get("weight_decay"),
                "lr": group.get("lr"),
            }
        )
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "groups_sha256": canonical_hash(groups),
        "group_count": len(groups),
    }


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


@torch.no_grad()
def evaluate(model, dataset, config, device):
    loader = DataLoader(
        dataset,
        batch_size=int(config["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )
    model.eval()
    correct1 = correct5 = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    precision = config.get("precision", "fp16" if config.get("amp", True) else "fp32")
    autocast_enabled = precision in ("fp16", "bf16")
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
            logits = model(images)
            loss_sum += float(criterion(logits, targets).item())
        predictions = logits.topk(5, dim=1).indices
        correct = predictions.eq(targets[:, None])
        correct1 += int(correct[:, :1].sum().item())
        correct5 += int(correct.sum().item())
        total += targets.numel()
    return {
        "count": total,
        "loss": loss_sum / total,
        "top1": 100.0 * correct1 / total,
        "top5": 100.0 * correct5 / total,
    }


def finite_gradients(model):
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )


def save_training_checkpoint(path, model, optimizer, scheduler, cursor, audit, history, timing_samples, config):
    atomic_torch_save(
        path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cursor": cursor,
            "audit": audit,
            "history": history,
            "timing_samples": timing_samples,
            "rng": capture_rng(),
            "config_sha256": canonical_hash(config),
        },
    )


def train(config, config_path: Path):
    if config["mode"] not in MODES:
        raise ValueError(f"invalid mode {config['mode']}")
    precision = config.get("precision", "fp16" if config.get("amp", True) else "fp32")
    if precision not in ("fp16", "bf16", "fp32"):
        raise ValueError(f"unsupported precision: {precision}")
    if float(config["fixed_loss_scale"]) <= 0:
        raise ValueError("fixed_loss_scale must be positive")
    device = torch.device(config.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal classification training requires CUDA")
    seed_all(int(config["seed"]))
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, source_audit = load_official_model(config, device)
    rand_schedule, rand_schedule_audit = load_rand_schedule(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    total_updates = int(config["planned_updates"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=float(config.get("min_lr", 0.0))
    )
    train_rows = read_rows(Path(config["train_index"]))
    selection_rows = read_rows(Path(config["selection_index"]))
    train_transform, eval_transform = transforms_for(int(config["image_size"]))
    root = Path(config["dataset_root"]).resolve()
    train_dataset = IndexedImageDataset(root, train_rows, train_transform, int(config["seed"]))
    selection_dataset = IndexedImageDataset(root, selection_rows, eval_transform, int(config["seed"]))
    audit = {
        "attempted_steps": 0,
        "effective_updates": 0,
        "scheduler_steps": 0,
        "amp_skips": 0,
        "nan_inf": 0,
        "zero_updates": 0,
        "data_order_sha256": [],
        "mode_update_counts": {"DDDD": 0, "SSSS": 0} if rand_schedule is not None else {},
    }
    history = []
    timing_samples = {"e2e": [], "gpu": [], "loader": []}
    cursor = {"epoch": 0, "batch": 0}
    checkpoint_path = output / "latest.pth"
    resume_path = Path(config["resume"]) if config.get("resume") else (
        checkpoint_path if checkpoint_path.is_file() else None
    )
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        if checkpoint["config_sha256"] != canonical_hash(config):
            raise RuntimeError("resume config hash differs from current frozen config")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        cursor = checkpoint["cursor"]
        audit = checkpoint["audit"]
        if rand_schedule is not None:
            audit.setdefault("mode_update_counts", {"DDDD": 0, "SSSS": 0})
        history = checkpoint["history"]
        timing_samples = checkpoint.get("timing_samples", timing_samples)
        restore_rng(checkpoint["rng"])
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.1)))
    e2e_times = timing_samples["e2e"]
    gpu_times = timing_samples["gpu"]
    loader_times = timing_samples["loader"]
    loss_scale = float(config["fixed_loss_scale"]) if precision == "fp16" else 1.0
    autocast_enabled = precision in ("fp16", "bf16")
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    torch.cuda.reset_peak_memory_stats(device)
    epoch = int(cursor["epoch"])
    while audit["effective_updates"] < total_updates and epoch < int(config["max_epochs"]):
        start_batch = int(cursor["batch"]) if epoch == int(cursor["epoch"]) else 0
        train_dataset.set_epoch(epoch)
        full_sampler = FrozenBatchSampler(
            train_rows, int(config["data_order_seed"]), epoch, int(config["batch_size"]), 0
        )
        sampler = FrozenBatchSampler(
            train_rows, int(config["data_order_seed"]), epoch, int(config["batch_size"]), start_batch
        )
        order = [train_rows[index]["relative_path"] for batch in full_sampler.batches for index in batch]
        order_hash = canonical_hash(order)
        if len(audit["data_order_sha256"]) <= epoch:
            audit["data_order_sha256"].append(order_hash)
        elif audit["data_order_sha256"][epoch] != order_hash:
            raise RuntimeError("data order changed across resume")
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=int(config["num_workers"]),
            pin_memory=True,
        )
        model.train()
        epoch_loss = 0.0
        seen = 0
        previous_end = time.perf_counter()
        for local_batch, (images, targets, _) in enumerate(loader, start=start_batch):
            if audit["effective_updates"] >= total_updates:
                break
            e2e_start = time.perf_counter()
            loader_times.append(e2e_start - previous_end)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            runtime_mode = None
            if rand_schedule is not None:
                runtime_mode = rand_schedule[audit["effective_updates"]]
                set_runtime_mode(model, runtime_mode)
            trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            parameter_versions = [parameter._version for parameter in trainable_parameters]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            audit["attempted_steps"] += 1
            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
                logits = model(images)
                loss = criterion(logits, targets)
            if not bool(torch.isfinite(loss).item()):
                audit["nan_inf"] += 1
                raise RuntimeError("nonfinite loss")
            (loss * loss_scale).backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(loss_scale)
            if not finite_gradients(model):
                audit["nan_inf"] += 1
                raise RuntimeError("nonfinite fixed-scale gradient")
            clip = float(config.get("clip_grad", 5.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            audit["effective_updates"] += 1
            audit["scheduler_steps"] += 1
            if not any(
                parameter._version > version
                for parameter, version in zip(trainable_parameters, parameter_versions)
            ):
                audit["zero_updates"] += 1
                raise RuntimeError("zero optimizer update")
            if runtime_mode is not None:
                audit["mode_update_counts"][runtime_mode] += 1
            end_event.record()
            torch.cuda.synchronize(device)
            gpu_times.append(start_event.elapsed_time(end_event) / 1000.0)
            e2e_times.append(time.perf_counter() - e2e_start)
            epoch_loss += float(loss.item()) * targets.numel()
            seen += targets.numel()
            cursor = {"epoch": epoch, "batch": local_batch + 1}
            interval = int(config.get("checkpoint_interval_updates", 500))
            if STOP_REQUESTED or audit["effective_updates"] % interval == 0:
                save_training_checkpoint(
                    checkpoint_path, model, optimizer, scheduler, cursor, audit, history, timing_samples, config
                )
            if STOP_REQUESTED:
                summary = {
                    "status": "PAUSED_BY_SIGNAL",
                    "cursor": cursor,
                    "audit": audit,
                    "checkpoint": str(checkpoint_path),
                }
                atomic_json(output / "summary.json", summary)
                return summary
            previous_end = time.perf_counter()
        selection = evaluate(model, selection_dataset, config, device)
        history.append(
            {
                "epoch": epoch,
                "effective_updates": audit["effective_updates"],
                "train_loss": epoch_loss / max(1, seen),
                "selection": selection,
            }
        )
        epoch += 1
        cursor = {"epoch": epoch, "batch": 0}
        save_training_checkpoint(
            checkpoint_path, model, optimizer, scheduler, cursor, audit, history, timing_samples, config
        )
        atomic_json(output / "history.json", history)
    clean = (
        audit["attempted_steps"] == audit["effective_updates"] == audit["scheduler_steps"]
        and audit["amp_skips"] == audit["nan_inf"] == audit["zero_updates"] == 0
    )
    status = "PASS" if clean and audit["effective_updates"] == total_updates else "FAIL"
    final_path = output / "final.pth"
    save_training_checkpoint(final_path, model, optimizer, scheduler, cursor, audit, history, timing_samples, config)
    # The epoch-end ``latest`` and the final artifact describe the same
    # completed state.  Atomically replace latest with a hard-link alias so a
    # completed run remains resumable without consuming checkpoint space twice.
    link_tmp = output / "latest.pth.linktmp"
    link_tmp.unlink(missing_ok=True)
    os.link(final_path, link_tmp)
    os.replace(link_tmp, checkpoint_path)
    timing = {
        "e2e_p50_seconds": statistics.median(e2e_times) if e2e_times else None,
        "e2e_p95_seconds": percentile(e2e_times, 0.95),
        "gpu_step_p50_seconds": statistics.median(gpu_times) if gpu_times else None,
        "gpu_step_p95_seconds": percentile(gpu_times, 0.95),
        "dataloader_p50_seconds": statistics.median(loader_times) if loader_times else None,
        "dataloader_p95_seconds": percentile(loader_times, 0.95),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    summary = {
        "status": status,
        "task": "ImageNet-100 fine-tuning external validation",
        "mode": config["mode"],
        "seed": config["seed"],
        "audit": audit,
        "timing": timing,
        "latest_selection": history[-1]["selection"] if history else None,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "source": source_audit,
        "rand_schedule": rand_schedule_audit,
        "scheduler_context": config.get("scheduler_context", "serial"),
        "parameter_signature": parameter_signature(model, optimizer),
        "config": str(config_path.resolve()),
        "config_sha256": canonical_hash(config),
        "dataset_index_hashes": {
            "train": sha256_file(Path(config["train_index"])),
            "selection": sha256_file(Path(config["selection_index"])),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_uuid": os.environ.get("NVIDIA_VISIBLE_DEVICES", "not_exposed_by_runtime"),
            "fixed_loss_scale": loss_scale,
            "precision": precision,
        },
    }
    atomic_json(output / "summary.json", summary)
    if status != "PASS":
        raise SystemExit(2)
    return summary


def evaluate_only(config, config_path: Path):
    device = torch.device(config.get("device", "cuda"))
    model, source_audit = load_official_model(config, device)
    checkpoint_path = Path(config["evaluation_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    _, eval_transform = transforms_for(int(config["image_size"]))
    rows = read_rows(Path(config["official_val_index"]))
    dataset = IndexedImageDataset(
        Path(config["dataset_root"]).resolve(), rows, eval_transform, int(config["seed"])
    )
    result = evaluate(model, dataset, config, device)
    payload = {
        "status": "PASS",
        "official_val_post_frozen": True,
        "mode": config["mode"],
        "seed": config["seed"],
        "metrics": result,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "official_val_index_sha256": sha256_file(Path(config["official_val_index"])),
        "source": source_audit,
        "config": str(config_path.resolve()),
    }
    output = Path(config["output"]).resolve()
    atomic_json(output / "official_val.json", payload)
    return payload


def handle_signal(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    result = train(config, args.config) if args.action == "train" else evaluate_only(config, args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
