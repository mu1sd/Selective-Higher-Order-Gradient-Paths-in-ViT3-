#!/usr/bin/env python3
"""Audited single-GPU dense-prediction fine-tuning with ViT3-T + UPerNet.

The same implementation is used for LoveDA and the prepared 128x256
Cityscapes derivative so that mode comparisons keep an identical optimizer,
precision path, sampler, checkpoint format, and metric implementation.
"""

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
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, Sampler

from training.classification import (
    STOP_REQUESTED,
    atomic_json,
    atomic_torch_save,
    canonical_hash,
    capture_rng,
    percentile,
    restore_rng,
    seed_all,
    sha256_file,
)


MODES = ("FFFF", "DDDD", "SSSS", "OOOO", "RAND", "GSHPS_AUTO", "GSHPS_SENS", "GSHPS_50", "RAND_PATH")
LOCAL_STOP_REQUESTED = False


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_relative_path", "mask_relative_path", "stem"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid segmentation index: {path}")
    for row in rows:
        row.setdefault("domain", "all")
    return rows


def safe_path(root: Path, relative: str):
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"index escapes dataset root: {path}") from exc
    return path


def map_mask(mask, dataset_type):
    if dataset_type == "cityscapes_processed19":
        mapped = np.asarray(mask, dtype=np.uint8).copy()
        invalid = ~((mapped <= 18) | (mapped == 255))
        if bool(invalid.any()):
            raise ValueError(f"prepared Cityscapes mask has invalid labels: {np.unique(mapped[invalid]).tolist()}")
        return mapped
    if dataset_type != "loveda":
        raise ValueError(f"unsupported segmentation dataset_type: {dataset_type}")
    mapped = np.full(mask.shape, 255, dtype=np.uint8)
    for value in range(1, 8):
        mapped[mask == value] = value - 1
    return mapped


class IndexedSegmentation(Dataset):
    def __init__(self, root: Path, rows, config, seed: int, training: bool):
        self.root = root
        self.rows = rows
        self.dataset_type = config.get("dataset_type", "loveda")
        self.target_height = int(config.get("target_height", config["crop_size"]))
        self.target_width = int(config.get("target_width", config["crop_size"]))
        self.seed = seed
        self.training = training
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = safe_path(self.root, row["image_relative_path"])
        mask_path = safe_path(self.root, row["mask_relative_path"])
        with Image.open(image_path) as value:
            image = value.convert("RGB")
        with Image.open(mask_path) as value:
            mask = value.convert("L")
        local_seed = int(
            hashlib.sha256(
                f"{self.seed}:{self.epoch}:{row['domain']}:{row['stem']}".encode()
            ).hexdigest()[:16],
            16,
        )
        generator = random.Random(local_seed)
        if self.training:
            low, high = (0.5, 2.0) if self.dataset_type == "loveda" else (0.75, 1.25)
            ratio = generator.uniform(low, high)
            base_height, base_width = image.height, image.width
            scaled_height = max(self.target_height, int(round(base_height * ratio)))
            scaled_width = max(self.target_width, int(round(base_width * ratio)))
            image = image.resize((scaled_width, scaled_height), Image.Resampling.BILINEAR)
            mask = mask.resize((scaled_width, scaled_height), Image.Resampling.NEAREST)
            left = generator.randint(0, scaled_width - self.target_width)
            top = generator.randint(0, scaled_height - self.target_height)
            box = (left, top, left + self.target_width, top + self.target_height)
            image = image.crop(box)
            mask = mask.crop(box)
            if generator.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(generator.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(generator.uniform(0.9, 1.1))
        image_array = np.asarray(image, dtype=np.float32).copy()
        mask_array = map_mask(np.asarray(mask, dtype=np.uint8), self.dataset_type)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array.astype(np.int64))
        meta = {
            "domain": row["domain"],
            "stem": row["stem"],
            "img_path": str(image_path),
            "seg_map_path": str(mask_path),
        }
        return image_tensor, mask_tensor, meta


class FrozenBatchSampler(Sampler):
    def __init__(self, rows, seed, epoch, batch_size, start_batch=0):
        ranked = sorted(
            range(len(rows)),
            key=lambda index: hashlib.sha256(
                f"{seed}:{epoch}:{rows[index]['domain']}:{rows[index]['stem']}".encode()
            ).hexdigest(),
        )
        batches = [
            ranked[offset : offset + batch_size]
            for offset in range(0, len(ranked), batch_size)
            if len(ranked[offset : offset + batch_size]) == batch_size
        ]
        self.full_order = ranked
        self.batches = batches[start_batch:]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate(batch):
    images, masks, metadata = zip(*batch)
    return list(images), list(masks), list(metadata)


def make_samples(masks, metadata):
    from mmengine.structures import PixelData
    from mmseg.structures import SegDataSample

    samples = []
    for mask, meta in zip(masks, metadata):
        height, width = mask.shape
        sample = SegDataSample(
            metainfo={
                **meta,
                "ori_shape": (height, width),
                "img_shape": (height, width),
                "pad_shape": (height, width),
            }
        )
        sample.gt_sem_seg = PixelData(data=mask.unsqueeze(0))
        samples.append(sample)
    return samples


def load_model(config, device):
    mmseg_root = Path(config["mmseg_root"]).resolve()
    sys.path.insert(0, str(mmseg_root))
    from mmengine.config import ConfigDict
    from mmengine.registry import init_default_scope
    from mmseg.models import build_segmentor

    init_default_scope("mmseg")

    model_cfg = dict(
        type="EncoderDecoder",
        data_preprocessor=dict(
            type="SegDataPreProcessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=False,
            pad_val=0,
            seg_pad_val=255,
            size=(
                int(config.get("target_height", config["crop_size"])),
                int(config.get("target_width", config["crop_size"])),
            ),
        ),
        backbone=dict(
            type="vittt_tiny",
            img_size=int(config["crop_size"]),
            drop_path_rate=float(config.get("drop_path_rate", 0.2)),
            gradient_mode=(config.get("runtime_initial_mode", "DDDD") if config["mode"] == "RAND" else ("PATH" if config["mode"] in ("GSHPS_AUTO", "GSHPS_SENS", "GSHPS_50", "RAND_PATH") else config["mode"])),
            use_checkpoint=bool(config.get("use_checkpoint", False)),
            init_cfg=None,
        ),
        decode_head=dict(
            type="UPerHead",
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=int(config.get("num_classes", 7)),
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        ),
        auxiliary_head=dict(
            type="FCNHead",
            in_channels=320,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=int(config.get("num_classes", 7)),
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
        ),
        train_cfg=ConfigDict(),
        test_cfg=ConfigDict(mode="whole"),
    )
    seed_all(int(config["seed"]))
    model = build_segmentor(model_cfg)
    checkpoint_path = Path(config["pretrained_checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    missing_backbone = [
        key
        for key in incompatible.missing_keys
        if key.startswith("backbone.") and "rope.rotations" not in key
    ]
    if missing_backbone:
        raise RuntimeError(f"segmentation checkpoint missing backbone keys: {missing_backbone[:20]}")
    from gshps.masks import PATH_MODES, apply_segmentation_mask, load_mask
    initial_mode = config.get("runtime_initial_mode", "DDDD") if config["mode"] == "RAND" else ("PATH" if config["mode"] in PATH_MODES else config["mode"])
    mode_count = model.backbone.set_gradient_mode(initial_mode)
    mask_audit = None
    if config["mode"] in PATH_MODES:
        mask_payload, mask_audit = load_mask(config)
        mode_count = apply_segmentation_mask(model, mask_payload["entries"])
    return model.to(device), {
        "mmseg_root": str(mmseg_root),
        "mmseg_commit": config["mmseg_commit"],
        "official_commit": config["official_commit"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "compatible_key_count": len(compatible),
        "missing_head_or_rope_keys": incompatible.missing_keys,
        "ignored_checkpoint_keys": len(state) - len(compatible),
        "ttt_block_count": mode_count,
        "use_checkpoint": bool(config.get("use_checkpoint", False)),
        "runtime_initial_mode": initial_mode,
        "path_mask": mask_audit,
        "upstream_scale_state_fix": "forward mutation removed; fixed at upstream long-run value 1/3",
    }


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


def optimizer_for(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(token in name for token in ("absolute_pos_embed", "relative_position_bias_table", "norm")):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.999),
    )
    from mmengine.optim import AmpOptimWrapper, OptimWrapper

    precision = config.get("precision", "fp16")
    wrapper_type = AmpOptimWrapper if precision == "fp16" else OptimWrapper
    kwargs = {
        "optimizer": optimizer,
        "clip_grad": dict(max_norm=float(config.get("clip_grad", 1.0)), norm_type=2),
    }
    if precision == "fp16":
        kwargs["loss_scale"] = float(config["fixed_loss_scale"])
    wrapper = wrapper_type(**kwargs)
    return optimizer, wrapper


def schedule_factor(step, total, warmup):
    if step < warmup:
        return max(1e-6, (step + 1) / max(1, warmup))
    return max(0.0, 1.0 - (step - warmup) / max(1, total - warmup))


def parse_loss(model, losses):
    loss, log_vars = model.parse_losses(losses)
    return loss, {key: float(value.detach().cpu()) for key, value in log_vars.items()}


@torch.no_grad()
def evaluate(model, dataset, config, device, qualitative_dir=None, qualitative_limit=0):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        collate_fn=collate,
    )
    num_classes = int(config.get("num_classes", 7))
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)
    domain_names = sorted({row.get("domain", "all") for row in dataset.rows})
    domain_i = {name: torch.zeros(num_classes, dtype=torch.float64) for name in domain_names}
    domain_u = {name: torch.zeros(num_classes, dtype=torch.float64) for name in domain_names}
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    qualitative_dir = Path(qualitative_dir).resolve() if qualitative_dir else None
    if qualitative_dir:
        qualitative_dir.mkdir(parents=True, exist_ok=True)
    qualitative_manifest = []
    model.eval()
    for images, masks, metadata in loader:
        samples = make_samples(masks, metadata)
        data = model.data_preprocessor({"inputs": images, "data_samples": samples}, training=False)
        predictions = model._run_forward(data, mode="predict")
        for prediction, target, meta in zip(predictions, masks, metadata):
            pred = prediction.pred_sem_seg.data.squeeze(0).cpu()
            if pred.shape != target.shape:
                pred = F.interpolate(
                    pred[None, None].float(), size=target.shape, mode="nearest"
                ).squeeze().long()
            valid = target != 255
            valid_target = target[valid].long()
            valid_pred = pred[valid].long()
            encoded = valid_target * num_classes + valid_pred
            confusion += torch.bincount(
                encoded, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
            for category in range(num_classes):
                intersection = ((pred == category) & (target == category) & valid).sum()
                union = (((pred == category) | (target == category)) & valid).sum()
                intersections[category] += intersection
                unions[category] += union
                domain_i[meta["domain"]][category] += intersection
                domain_u[meta["domain"]][category] += union
            if qualitative_dir and len(qualitative_manifest) < int(qualitative_limit):
                sample_id = f"{len(qualitative_manifest):02d}_{meta['domain']}_{meta['stem']}"
                sample_path = qualitative_dir / f"{sample_id}.npz"
                image_uint8 = images[0].detach().cpu().permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
                target_uint8 = target.detach().cpu().clamp(0, 255).to(torch.uint8).numpy()
                pred_uint8 = pred.detach().cpu().clamp(0, 255).to(torch.uint8).numpy()
                error_uint8 = ((pred != target) & valid).to(torch.uint8).numpy()
                np.savez_compressed(
                    sample_path,
                    image=image_uint8,
                    target=target_uint8,
                    prediction=pred_uint8,
                    error=error_uint8,
                )
                qualitative_manifest.append({
                    "sample_id": sample_id,
                    "domain": meta["domain"],
                    "stem": meta["stem"],
                    "source_image": meta["img_path"],
                    "source_mask": meta["seg_map_path"],
                    "artifact": str(sample_path),
                    "artifact_sha256": sha256_file(sample_path),
                    "private_restricted_data": True,
                })
    iou = intersections / unions.clamp_min(1)
    present = unions > 0
    domain = {
        key: float((domain_i[key] / domain_u[key].clamp_min(1))[domain_u[key] > 0].mean() * 100)
        for key in domain_i
        if bool((domain_u[key] > 0).any())
    }
    payload = {
        "miou": float(iou[present].mean() * 100),
        "per_class_iou": [float(value * 100) if bool(flag) else None for value, flag in zip(iou, present)],
        "present_class_count": int(present.sum()),
        "domain_miou": domain,
        "intersection": intersections.tolist(),
        "union": unions.tolist(),
        "confusion_matrix_true_by_pred": confusion.tolist(),
    }
    if str(config.get("dataset_type", "")).startswith("cityscapes") and num_classes == 19:
        mapping_19_to_7 = torch.tensor(
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 6],
            dtype=torch.long,
        )
        confusion_7 = torch.zeros((7, 7), dtype=torch.int64)
        for true_19 in range(19):
            for pred_19 in range(19):
                confusion_7[mapping_19_to_7[true_19], mapping_19_to_7[pred_19]] += confusion[true_19, pred_19]
        intersection_7 = confusion_7.diag().to(torch.float64)
        union_7 = confusion_7.sum(0) + confusion_7.sum(1) - confusion_7.diag()
        present_7 = union_7 > 0
        iou_7 = intersection_7 / union_7.clamp_min(1)
        payload["secondary_7class"] = {
            "status": "PASS",
            "miou": float(iou_7[present_7].mean() * 100),
            "per_class_iou": [
                float(value * 100) if bool(flag) else None
                for value, flag in zip(iou_7, present_7)
            ],
            "confusion_matrix_true_by_pred": confusion_7.tolist(),
            "mapping_19_to_7": mapping_19_to_7.tolist(),
            "evaluation_scope": "pixels valid under the frozen segmentation_19 primary task; source -1 remains ignore=255",
            "scientific_role": "secondary sensitivity only",
        }
    if qualitative_dir:
        atomic_json(
            qualitative_dir / "manifest.json",
            {
                "status": "PASS",
                "count": len(qualitative_manifest),
                "items": qualitative_manifest,
                "distribution_rule": "PRIVATE_RESTRICTED; exclude from public evidence bundle",
            },
        )
        payload["qualitative_private_manifest"] = str(qualitative_dir / "manifest.json")
    return payload


def save_checkpoint(path, model, wrapper, scheduler, cursor, audit, history, timing_samples, config):
    atomic_torch_save(
        path,
        {
            "model": model.state_dict(),
            "optim_wrapper": wrapper.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cursor": cursor,
            "audit": audit,
            "history": history,
            "timing_samples": timing_samples,
            "rng": capture_rng(),
            "config_sha256": canonical_hash(config),
        },
    )


def train(config, config_path):
    if config["mode"] not in MODES:
        raise ValueError(f"invalid mode: {config['mode']}")
    precision = config.get("precision", "fp16")
    if precision not in ("fp16", "bf16", "fp32"):
        raise ValueError(f"unsupported precision: {precision}")
    device = torch.device(config.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal segmentation training requires CUDA")
    seed_all(int(config["seed"]))
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, source_audit = load_model(config, device)
    rand_schedule, rand_schedule_audit = load_rand_schedule(config)
    optimizer, wrapper = optimizer_for(model, config)
    total_updates = int(config["planned_updates"])
    warmup = int(config.get("warmup_updates", min(1500, total_updates // 10)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: schedule_factor(step, total_updates, warmup)
    )
    rows = read_rows(Path(config["train_index"]))
    selection_rows = read_rows(Path(config["selection_index"]))
    root = Path(config["dataset_root"]).resolve()
    dataset = IndexedSegmentation(root, rows, config, int(config["seed"]), True)
    selection = IndexedSegmentation(root, selection_rows, config, int(config["seed"]), False)
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
    resume = Path(config["resume"]) if config.get("resume") else (
        checkpoint_path if checkpoint_path.is_file() else None
    )
    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        if checkpoint["config_sha256"] != canonical_hash(config):
            raise RuntimeError("resume config hash differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        wrapper.load_state_dict(checkpoint["optim_wrapper"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        cursor, audit, history = checkpoint["cursor"], checkpoint["audit"], checkpoint["history"]
        if rand_schedule is not None:
            audit.setdefault("mode_update_counts", {"DDDD": 0, "SSSS": 0})
        timing_samples = checkpoint.get("timing_samples", timing_samples)
        restore_rng(checkpoint["rng"])
    e2e_times = timing_samples["e2e"]
    gpu_times = timing_samples["gpu"]
    loader_times = timing_samples["loader"]
    torch.cuda.reset_peak_memory_stats(device)
    epoch = int(cursor["epoch"])
    last_logs = {}
    # A process can be interrupted after the last optimizer update but before
    # the epoch-end selection evaluation.  Resume must finish that atomic
    # reporting boundary instead of emitting a metric-less PASS.
    if audit["effective_updates"] >= total_updates and not history:
        history.append(
            {
                "epoch": epoch,
                "effective_updates": audit["effective_updates"],
                "selection": evaluate(model, selection, config, device),
                "last_losses": {},
                "recovered_epoch_end_evaluation": True,
            }
        )
        atomic_json(output / "history.json", history)
    while audit["effective_updates"] < total_updates and epoch < int(config["max_epochs"]):
        start_batch = int(cursor["batch"]) if epoch == int(cursor["epoch"]) else 0
        dataset.set_epoch(epoch)
        sampler = FrozenBatchSampler(
            rows, int(config["data_order_seed"]), epoch, int(config["batch_size"]), start_batch
        )
        full_sampler = FrozenBatchSampler(
            rows, int(config["data_order_seed"]), epoch, int(config["batch_size"]), 0
        )
        order_hash = canonical_hash(
            [f"{rows[index]['domain']}:{rows[index]['stem']}" for index in full_sampler.full_order]
        )
        if len(audit["data_order_sha256"]) <= epoch:
            audit["data_order_sha256"].append(order_hash)
        elif audit["data_order_sha256"][epoch] != order_hash:
            raise RuntimeError("segmentation data order changed across resume")
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(config["num_workers"]),
            pin_memory=True,
            collate_fn=collate,
        )
        previous_end = time.perf_counter()
        for batch_index, (images, masks, metadata) in enumerate(loader, start=start_batch):
            if audit["effective_updates"] >= total_updates:
                break
            e2e_start = time.perf_counter()
            loader_times.append(e2e_start - previous_end)
            samples = make_samples(masks, metadata)
            runtime_mode = None
            if rand_schedule is not None:
                runtime_mode = rand_schedule[audit["effective_updates"]]
                if model.backbone.set_gradient_mode(runtime_mode) <= 0:
                    raise RuntimeError(f"no official TTT blocks accepted runtime mode {runtime_mode}")
            model.train()
            data = model.data_preprocessor({"inputs": images, "data_samples": samples}, training=True)
            # A single parameter can remain bitwise unchanged at a small
            # learning rate even though the optimizer performed a valid
            # update.  Tensor version counters advance on the optimizer's
            # in-place parameter writes and therefore distinguish a real
            # optimizer step from an AMP-scaler skip without depending on
            # the numerical delta of one arbitrarily selected tensor.
            trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            parameter_versions = [parameter._version for parameter in trainable_parameters]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            audit["attempted_steps"] += 1
            autocast_enabled = precision in ("fp16", "bf16")
            autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
            with wrapper.optim_context(model):
                with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
                    losses = model._run_forward(data, mode="loss")
                    loss, last_logs = parse_loss(model, losses)
            if not bool(torch.isfinite(loss).item()):
                audit["nan_inf"] += 1
                raise RuntimeError("nonfinite segmentation loss")
            scale_before = float(wrapper.loss_scaler.get_scale()) if hasattr(wrapper, "loss_scaler") else None
            wrapper.update_params(loss)
            if scale_before is not None and float(wrapper.loss_scaler.get_scale()) != scale_before:
                raise RuntimeError("fixed loss scale changed")
            optimizer_step_applied = any(
                parameter._version > version
                for parameter, version in zip(trainable_parameters, parameter_versions)
            )
            if not optimizer_step_applied:
                audit["amp_skips"] += 1
                audit["zero_updates"] += 1
                raise RuntimeError("segmentation AMP skip or zero update")
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                audit["nan_inf"] += 1
                raise RuntimeError("nonfinite segmentation parameter")
            scheduler.step()
            audit["effective_updates"] += 1
            audit["scheduler_steps"] += 1
            if runtime_mode is not None:
                audit["mode_update_counts"][runtime_mode] += 1
            end_event.record()
            torch.cuda.synchronize(device)
            gpu_times.append(start_event.elapsed_time(end_event) / 1000.0)
            e2e_times.append(time.perf_counter() - e2e_start)
            cursor = {"epoch": epoch, "batch": batch_index + 1}
            interval = int(config.get("checkpoint_interval_updates", 250))
            if LOCAL_STOP_REQUESTED or audit["effective_updates"] % interval == 0:
                save_checkpoint(checkpoint_path, model, wrapper, scheduler, cursor, audit, history, timing_samples, config)
            if LOCAL_STOP_REQUESTED:
                payload = {"status": "PAUSED_BY_SIGNAL", "cursor": cursor, "audit": audit}
                atomic_json(output / "summary.json", payload)
                return payload
            previous_end = time.perf_counter()
        selection_metrics = evaluate(model, selection, config, device)
        history.append(
            {
                "epoch": epoch,
                "effective_updates": audit["effective_updates"],
                "selection": selection_metrics,
                "last_losses": last_logs,
            }
        )
        epoch += 1
        cursor = {"epoch": epoch, "batch": 0}
        save_checkpoint(checkpoint_path, model, wrapper, scheduler, cursor, audit, history, timing_samples, config)
        atomic_json(output / "history.json", history)
    clean = (
        audit["attempted_steps"] == audit["effective_updates"] == audit["scheduler_steps"]
        and audit["amp_skips"] == audit["nan_inf"] == audit["zero_updates"] == 0
    )
    status = "PASS" if clean and audit["effective_updates"] == total_updates else "FAIL"
    final_path = output / "final.pth"
    save_checkpoint(final_path, model, wrapper, scheduler, cursor, audit, history, timing_samples, config)
    # Keep both stable names while storing the completed state only once.
    link_tmp = output / "latest.pth.linktmp"
    link_tmp.unlink(missing_ok=True)
    os.link(final_path, link_tmp)
    os.replace(link_tmp, checkpoint_path)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    group_payload = [
        [names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups
    ]
    summary = {
        "status": status,
        "task": config.get("task_name", "LoveDA semantic-segmentation external validation"),
        "dataset_type": config.get("dataset_type", "loveda"),
        "mode": config["mode"],
        "seed": config["seed"],
        "audit": audit,
        "latest_selection": history[-1]["selection"] if history else None,
        "timing": {
            "e2e_p50_seconds": statistics.median(e2e_times) if e2e_times else None,
            "e2e_p95_seconds": percentile(e2e_times, 0.95),
            "gpu_step_p50_seconds": statistics.median(gpu_times) if gpu_times else None,
            "gpu_step_p95_seconds": percentile(gpu_times, 0.95),
            "dataloader_p50_seconds": statistics.median(loader_times) if loader_times else None,
            "dataloader_p95_seconds": percentile(loader_times, 0.95),
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "source": source_audit,
        "rand_schedule": rand_schedule_audit,
        "scheduler_context": config.get("scheduler_context", "serial"),
        "optimizer_group_sha256": canonical_hash(group_payload),
        "config": str(Path(config_path).resolve()),
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
            "fixed_loss_scale": config["fixed_loss_scale"],
            "precision": precision,
        },
    }
    atomic_json(output / "summary.json", summary)
    if status != "PASS":
        raise SystemExit(2)
    return summary


def evaluate_only(config, config_path):
    device = torch.device(config.get("device", "cuda"))
    model, source_audit = load_model(config, device)
    checkpoint_path = Path(config["evaluation_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    rows = read_rows(Path(config["official_val_index"]))
    dataset = IndexedSegmentation(
        Path(config["dataset_root"]).resolve(),
        rows,
        config,
        int(config["seed"]),
        False,
    )
    metrics = evaluate(
        model,
        dataset,
        config,
        device,
        qualitative_dir=Path(config["output"]) / "qualitative_private",
        qualitative_limit=int(config.get("qualitative_sample_limit", 12)),
    )
    payload = {
        "status": "PASS",
        "official_val_post_frozen": True,
        "mode": config["mode"],
        "seed": config["seed"],
        "metrics": metrics,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "official_val_index_sha256": sha256_file(Path(config["official_val_index"])),
        "source": source_audit,
        "config": str(Path(config_path).resolve()),
    }
    atomic_json(Path(config["output"]) / "official_val.json", payload)
    return payload


def handle_signal(_signum, _frame):
    global LOCAL_STOP_REQUESTED
    LOCAL_STOP_REQUESTED = True


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
