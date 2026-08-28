#!/usr/bin/env python3
"""Pre-validation GSHPS calibration on two frozen train-core minibatches."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from campaign.io_utils import atomic_json
from gshps.masks import apply_classification_mask, apply_segmentation_mask, canonical_hash


BRANCHES = ("swiglu", "dwc")
# P2.1n: preserve the frozen logical batch but use one-sample shards.  This
# intentionally trades calibration-only wall time for a large physical-memory
# reserve on the 24 GiB card; sample-count weighting preserves the logical
# mean-loss and gradient definition.
CLASSIFICATION_CALIBRATION_MICROBATCH = 1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entries(blocks, keep=True):
    return [{"block_index": i, "swiglu": bool(keep), "dwc": bool(keep)} for i in range(blocks)]


def mask_modules(model, task):
    root = model if task == "imagenet100" else model.backbone
    return [m for m in root.modules() if getattr(m, "set_high_order_path_mask", None) is not None]


def apply(model, task, value):
    return apply_classification_mask(model, value) if task == "imagenet100" else apply_segmentation_mask(model, value)


def gradient_snapshot(model):
    result = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            # P2.1a: comparison-only evidence lives on host memory.  This is an
            # exact float32 copy and does not participate in optimization or
            # model execution; keeping it off-device preserves the frozen
            # P2.1d's 23,500 MiB ceiling is below 23 GiB (23,552 MiB).
            result.append(None if parameter.grad is None else parameter.grad.detach().float().cpu().clone())
    return result


def compare_gradient(model, baseline):
    dot = left = right = diff = 0.0
    for parameter, reference in zip((p for p in model.parameters() if p.requires_grad), baseline):
        if parameter.grad is None and reference is None:
            continue
        current = torch.zeros_like(reference) if parameter.grad is None else parameter.grad.detach().float().cpu()
        reference = torch.zeros_like(current) if reference is None else reference
        dot += float((current * reference).sum())
        left += float(reference.square().sum())
        right += float(current.square().sum())
        diff += float((reference - current).square().sum())
    cosine = dot / max(math.sqrt(left * right), 1e-30)
    return 1.0 - cosine, math.sqrt(diff) / max(math.sqrt(left), 1e-30)


def saved_bytes_context():
    seen = set(); total = {"bytes": 0}
    def pack(tensor):
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
        if key not in seen:
            seen.add(key); total["bytes"] += tensor.numel() * tensor.element_size()
        return tensor
    return total, torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor)


def activation_costs(modules):
    costs = {}
    handles = []
    for index, module in enumerate(modules):
        def hook(current, args, block=index):
            x = args[0]; b, n, c = x.shape; element = x.element_size(); d = c // current.num_heads
            costs[(block, "swiglu")] = int(element * (6 * b * n * c + 4 * current.w1.numel()))
            costs[(block, "dwc")] = int(element * (5 * b * n * d + 3 * current.w3.numel()))
        handles.append(module.register_forward_pre_hook(hook))
    return costs, handles


def classification_batches(config):
    from training.classification import FrozenBatchSampler, IndexedImageDataset, read_rows, transforms_for
    rows = read_rows(Path(config["train_index"])); train_transform, _ = transforms_for(int(config["image_size"]))
    dataset = IndexedImageDataset(Path(config["dataset_root"]).resolve(), rows, train_transform, 20260720)
    dataset.set_epoch(0)
    sampler = FrozenBatchSampler(rows, 20260720, 0, int(config["batch_size"]))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    batches = []
    for images, targets, paths in loader:
        # P2.1c: retain the immutable logical minibatch on host memory.  The
        # same 64 samples are transferred in thirty-two exact, weighted gradient
        # accumulation shards by run_pass, preventing a near-capacity peak.
        batches.append((images, targets, list(paths)))
        if len(batches) == 2: break
    return batches


def segmentation_batches(config):
    from training.segmentation import FrozenBatchSampler, IndexedSegmentation, collate, read_rows
    rows = read_rows(Path(config["train_index"])); dataset = IndexedSegmentation(Path(config["dataset_root"]).resolve(), rows, config, 20260720, True)
    dataset.set_epoch(0); sampler = FrozenBatchSampler(rows, 20260720, 0, int(config["batch_size"]))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate)
    batches = []
    for images, masks, metadata in loader:
        batches.append((images, masks, metadata))
        if len(batches) == 2: break
    return batches


def task_loss(model, task, batch):
    if task == "imagenet100":
        images, targets, _ = batch
        return F.cross_entropy(model(images), targets, label_smoothing=0.1)
    from training.segmentation import make_samples, parse_loss
    images, masks, metadata = batch; samples = make_samples(masks, metadata)
    data = model.data_preprocessor({"inputs": images, "data_samples": samples}, training=True)
    losses = model._run_forward(data, mode="loss"); loss, _ = parse_loss(model, losses)
    return loss


def run_pass(model, task, batch):
    model.zero_grad(set_to_none=True)
    total, context = saved_bytes_context()
    if task == "imagenet100":
        images, targets, _ = batch
        logical_size = int(images.shape[0]); loss_value = 0.0
        with context:
            for start in range(0, logical_size, CLASSIFICATION_CALIBRATION_MICROBATCH):
                stop = min(start + CLASSIFICATION_CALIBRATION_MICROBATCH, logical_size)
                shard_images = images[start:stop].cuda(non_blocking=True)
                shard_targets = targets[start:stop].cuda(non_blocking=True)
                shard_loss = F.cross_entropy(model(shard_images), shard_targets, label_smoothing=0.1)
                weight = (stop - start) / logical_size
                (shard_loss * weight).backward()
                loss_value += float(shard_loss.detach()) * weight
                del shard_images, shard_targets, shard_loss
                # Keep freed shard blocks in the allocator until this logical
                # pass finishes. Repeated allocator teardown/rebuild here was
                # associated with rising physical use and a CUDA handle assert.
                # The caller releases unused cache at pass boundaries.
        loss = loss_value
    else:
        with context:
            loss = task_loss(model, task, batch)
        loss.backward()
        loss = float(loss.detach())
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise RuntimeError("non-finite calibration gradient")
    return float(loss), total["bytes"]


def choose(rows, coverage, ranking_key="score"):
    """Coverage selection with explicit 0%/100% boundaries and no validation."""
    ranked = sorted(rows, key=lambda row: (-row[ranking_key], row["block_index"], row["branch"]))
    total = sum(row["importance"] for row in ranked); cumulative = 0.0; selected = []
    if total <= 1e-15:
        return set()
    for row in ranked:
        selected.append((row["block_index"], row["branch"])); cumulative += row["importance"]
        if cumulative >= coverage * total: break
    return set(selected)


def jaccard(left, right):
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def build_mask(task, rows, selected, rule):
    block_count = 1 + max(row["block_index"] for row in rows)
    value = entries(block_count, False)
    for block, branch in selected: value[block][branch] = True
    return {"schema": 1, "task": task, "rule": rule, "entries": value, "kept_paths": len(selected), "total_paths": len(rows), "entries_sha256": canonical_hash(value)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--task", choices=("imagenet100", "loveda"), required=True); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(); base_config = json.loads(args.config.read_text(encoding="utf-8")); release = Path(__file__).resolve().parents[1]
    base_config.update({"mode": "FFFF", "use_checkpoint": False,
                   "official_root": str(release / "source" / "ViTTT"), "mmseg_root": str(release / "private_mmseg"),
                   "p2_method_release": str(release)})
    device = torch.device("cuda"); calibration_seeds=(11,23,41); total_started=time.perf_counter(); observations={}; fixed=[]; unit_rows=[]; source_by_seed={}; block_count=None
    batches = classification_batches(base_config) if args.task == "imagenet100" else segmentation_batches(base_config)
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=True); input_config_sha256=sha(args.config)

    def record_unit(unit):
        nonlocal block_count, observations
        if unit.get("status") != "PASS" or unit.get("task") != args.task or unit.get("input_config_sha256") != input_config_sha256:
            raise RuntimeError("calibration unit checkpoint contract mismatch")
        unit_blocks=int(unit["block_count"])
        if block_count is None:
            block_count=unit_blocks; observations={(i,b):[] for i in range(block_count) for b in BRANCHES}
        if unit_blocks != block_count or len(unit["rows"]) != 2*block_count:
            raise RuntimeError("calibration unit checkpoint shape mismatch")
        seed=int(unit["seed"]); batch_index=int(unit["batch"]); selected={tuple(value) for value in unit["selected"]}
        for item in unit["rows"]:
            observations[(item["block_index"],item["branch"])].append({"seed":seed,"batch":batch_index,**item})
        unit_rows.append({"seed":seed,"batch":batch_index,"rows":unit["rows"],"selected":selected})
        fixed.append(unit["fixed"]); source_by_seed[seed]=unit["source_audit"]

    for calibration_seed in calibration_seeds:
        config=dict(base_config); config.update({"seed":calibration_seed,"head_seed":calibration_seed,"data_order_seed":calibration_seed})
        for batch_index,batch in enumerate(batches):
            unit_path=out/f"{args.task}_calibration_unit_seed{calibration_seed}_batch{batch_index}.json"
            if unit_path.is_file():
                unit=json.loads(unit_path.read_text(encoding="utf-8")); record_unit(unit)
                print(json.dumps({"event":"calibration_unit_resumed","task":args.task,"seed":calibration_seed,"batch":batch_index,"path":str(unit_path)}),flush=True)
                continue
            if args.task == "imagenet100":
                from training.classification import load_official_model, seed_all
                seed_all(calibration_seed); model, source = load_official_model(config, device)
            else:
                from training.segmentation import load_model, seed_all
                seed_all(calibration_seed); model, source = load_model(config, device)
            source_audit={"seed":calibration_seed,"source":source,"config_sha256":canonical_hash(config)}; model.eval(); modules=mask_modules(model,args.task)
            if block_count is None:
                block_count=len(modules); observations={(i,b):[] for i in range(block_count) for b in BRANCHES}
            if len(modules)!=block_count or block_count<=0: raise RuntimeError("calibration block count is absent or seed-dependent")
            proxy,handles=activation_costs(modules); unit_started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device); full=entries(block_count,True); apply(model,args.task,full); full_loss,full_saved=run_pass(model,args.task,batch); baseline=gradient_snapshot(model); model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); raw=[]
            print(json.dumps({"event":"calibration_unit_baseline_complete","task":args.task,"seed":calibration_seed,"batch":batch_index,"allocated_mb":torch.cuda.memory_allocated(device)/1024**2,"reserved_mb":torch.cuda.memory_reserved(device)/1024**2}), flush=True)
            identity=batch[2] if args.task=="imagenet100" else [item["stem"] for item in batch[2]]
            for block in range(block_count):
                for branch in BRANCHES:
                    candidate=entries(block_count,True); candidate[block][branch]=False; apply(model,args.task,candidate); loss,saved=run_pass(model,args.task,batch); direction,magnitude=compare_gradient(model,baseline); marginal=full_saved-saved; method="saved_tensor_marginal"
                    if marginal<=0: marginal=proxy.get((block,branch),1); method="branch_activation_bytes_fallback"
                    raw.append({"block_index":block,"branch":branch,"direction":direction,"magnitude":magnitude,"loss_abs_diff":abs(loss-full_loss),"cost_bytes":int(marginal),"cost_method":method})
                    model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    print(json.dumps({"event":"calibration_candidate_complete","task":args.task,"seed":calibration_seed,"batch":batch_index,"block":block,"branch":branch,"completed_candidates":len(raw),"total_candidates":2*block_count,"allocated_mb":torch.cuda.memory_allocated(device)/1024**2,"reserved_mb":torch.cuda.memory_reserved(device)/1024**2}), flush=True)
            dsum=sum(x["direction"] for x in raw); nsum=sum(x["magnitude"] for x in raw); cmean=statistics.mean(x["cost_bytes"] for x in raw)
            if cmean<=0: raise RuntimeError("P2.1 path-cost normalization denominator is zero")
            normalized=[]
            for item in raw:
                item["direction_normalized"]=0.0 if dsum<=1e-15 else item["direction"]/dsum; item["magnitude_normalized"]=0.0 if nsum<=1e-15 else item["magnitude"]/nsum; item["importance"]=0.5*item["direction_normalized"]+0.5*item["magnitude_normalized"]; item["cost_normalized"]=item["cost_bytes"]/cmean; item["score"]=item["importance"]/(item["cost_normalized"]+1e-12); normalized.append(item)
            selected=choose(normalized,.90,"score")
            fixed_record={"seed":calibration_seed,"batch":batch_index,"sample_ids":identity,"sample_ids_sha256":canonical_hash(identity),"full_loss":full_loss,"full_saved_tensor_bytes":full_saved,"direction_sum":dsum,"magnitude_sum":nsum,"mean_cost_bytes":cmean,"elapsed_seconds":time.perf_counter()-unit_started,"peak_allocated_mb":torch.cuda.max_memory_allocated(device)/1024**2,"peak_reserved_mb":torch.cuda.max_memory_reserved(device)/1024**2}
            unit={"schema":1,"status":"PASS","amendment":"P2.1o","task":args.task,"seed":calibration_seed,"batch":batch_index,"input_config_sha256":input_config_sha256,"block_count":block_count,"rows":normalized,"selected":[list(value) for value in sorted(selected)],"fixed":fixed_record,"source_audit":source_audit,"official_validation_used":False}
            atomic_json(unit_path,unit); record_unit(unit)
            print(json.dumps({"event":"calibration_unit_checkpointed","task":args.task,"seed":calibration_seed,"batch":batch_index,"path":str(unit_path),"sha256":sha(unit_path)}),flush=True)
            for handle in handles: handle.remove()
            del baseline, model, modules, proxy, handles; gc.collect(); torch.cuda.empty_cache()
    source_audits=[source_by_seed[seed] for seed in calibration_seeds]
    rows=[]
    for (block,branch),values in observations.items():
        rows.append({"task":args.task,"block_index":block,"branch":branch,"direction_sensitivity":statistics.mean(v["direction"] for v in values),"magnitude_sensitivity":statistics.mean(v["magnitude"] for v in values),"direction_normalized":statistics.mean(v["direction_normalized"] for v in values),"magnitude_normalized":statistics.mean(v["magnitude_normalized"] for v in values),"importance":statistics.mean(v["importance"] for v in values),"importance_sd":statistics.stdev(v["importance"] for v in values),"cost_bytes":statistics.mean(v["cost_bytes"] for v in values),"cost_normalized":statistics.mean(v["cost_normalized"] for v in values),"score":statistics.mean(v["score"] for v in values),"score_sd":statistics.stdev(v["score"] for v in values),"cost_methods":"+".join(sorted({v["cost_method"] for v in values})),"max_forward_loss_abs_diff":max(v["loss_abs_diff"] for v in values)})
    auto_selected=choose(rows,.90,"score"); sensitivity_selected=choose(rows,.90,"importance"); half_selected=set((r["block_index"],r["branch"]) for r in sorted(rows,key=lambda r:(-r["score"],r["block_index"],r["branch"]))[:len(rows)//2])
    seed_masks=[]
    for seed in calibration_seeds:
        seed_values=[]
        for row in rows:
            values=[v for v in observations[(row["block_index"],row["branch"])] if v["seed"]==seed]; seed_values.append({"block_index":row["block_index"],"branch":row["branch"],"importance":statistics.mean(v["importance"] for v in values),"score":statistics.mean(v["score"] for v in values)})
        seed_masks.append({"seed":seed,"selected":choose(seed_values,.90,"score")})
    unit_sets=[u["selected"] for u in unit_rows]; unit_j=[jaccard(unit_sets[i],unit_sets[j]) for i in range(len(unit_sets)) for j in range(i+1,len(unit_sets))]; seed_sets=[x["selected"] for x in seed_masks]; seed_j=[jaccard(seed_sets[i],seed_sets[j]) for i in range(len(seed_sets)) for j in range(i+1,len(seed_sets))]
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    auto = build_mask(args.task, rows, auto_selected, {"name":"GSHPS-Auto","direction_normalization":"D/sum(D) within each seed×batch unit","magnitude_normalization":"N/sum(N) within each seed×batch unit","importance":"0.5*D_normalized+0.5*N_normalized","cost_normalization":"C/mean(C) within each seed×batch unit","ranking":"importance/(cost_normalized+1e-12)","coverage":0.90,"allowed_boundaries":[0.0,1.0]})
    sensitivity = build_mask(args.task, rows, sensitivity_selected, {"name":"GSHPS-Sensitivity-only","importance":"same normalized 0.5/0.5 importance as GSHPS-Auto","cost_used":False,"coverage":0.90,"allowed_boundaries":[0.0,1.0]})
    half = build_mask(args.task, rows, half_selected, {"name":"GSHPS-50","kept_fraction":0.50})
    auto_path=out/f"{args.task}_gshps_auto_mask.json"; sensitivity_path=out/f"{args.task}_gshps_sensitivity_mask.json"; half_path=out/f"{args.task}_gshps_50_mask.json"; atomic_json(auto_path,auto); atomic_json(sensitivity_path,sensitivity); atomic_json(half_path,half)
    with (out/f"{args.task}_path_scores.csv").open("w",encoding="utf-8",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fixed_hashes={batch_index:sorted({x["sample_ids_sha256"] for x in fixed if x["batch"]==batch_index}) for batch_index in (0,1)}
    if any(len(values)!=1 for values in fixed_hashes.values()): raise RuntimeError("fixed calibration minibatch identity changed across initialization seeds")
    audit={"status":"PASS","amendment":"P2.1a+P2.1b+P2.1c","task":args.task,"calibration_initialization_seeds":list(calibration_seeds),"minibatches_per_seed":2,"calibration_unit_count":6,"block_count":block_count,"path_count":2*block_count,"fixed_train_core_batches":fixed,"same_fixed_train_core_minibatches_across_initializations":True,"fixed_minibatch_sha256s":[fixed_hashes[i][0] for i in (0,1)],"official_validation_used":False,"comparison_gradient_snapshot":{"dtype":"float32","device":"cpu","scientific_role":"comparison-only; never used by forward, optimizer, scheduler, or mask budget"},"classification_calibration_accumulation":{"logical_minibatch_size":int(base_config["batch_size"]) if args.task=="imagenet100" else None,"microbatch_size":CLASSIFICATION_CALIBRATION_MICROBATCH if args.task=="imagenet100" else None,"loss_weighting":"microbatch_sample_count/logical_minibatch_sample_count","sample_identity_and_logical_batch_unchanged":True},"source_audits":source_audits,"normalization":{"direction":"per-unit sum-to-one, or all-zero when its raw sum is <= epsilon","magnitude":"per-unit sum-to-one, or all-zero when its raw sum is <= epsilon","cost":"per-unit divide-by-mean","epsilon":1e-12},"budget_boundaries":{"minimum_kept_fraction":0.0,"maximum_kept_fraction":1.0,"zero_selected_if_total_importance_le":1e-15,"all_paths_permitted_when_coverage_requires_them":True},"mask_stability":{"unit_pairwise_jaccard_count":len(unit_j),"unit_pairwise_jaccard_mean":statistics.mean(unit_j),"unit_pairwise_jaccard_min":min(unit_j),"seed_pairwise_jaccard_count":len(seed_j),"seed_pairwise_jaccard_mean":statistics.mean(seed_j),"seed_pairwise_jaccard_min":min(seed_j),"seed_masks":[{"seed":x["seed"],"kept_paths":len(x["selected"]),"selected":sorted([list(p) for p in x["selected"]])} for x in seed_masks]},"calibration_resources":{"elapsed_seconds":time.perf_counter()-total_started,"peak_allocated_mb":max(x["peak_allocated_mb"] for x in fixed),"peak_reserved_mb":max(x["peak_reserved_mb"] for x in fixed)},"auto_mask":{"path":str(auto_path),"sha256":sha(auto_path),"kept_paths":auto["kept_paths"]},"sensitivity_only_mask":{"path":str(sensitivity_path),"sha256":sha(sensitivity_path),"kept_paths":sensitivity["kept_paths"]},"gshps_50_mask":{"path":str(half_path),"sha256":sha(half_path),"kept_paths":half["kept_paths"]},"max_forward_loss_abs_diff":max(r["max_forward_loss_abs_diff"] for r in rows)}
    audit["amendment"] = "P2.1a+P2.1b+P2.1c+P2.1n+P2.1o"
    audit["calibration_resources"]["elapsed_seconds"] = sum(item["elapsed_seconds"] for item in fixed)
    audit["calibration_resources"]["current_invocation_elapsed_seconds"] = time.perf_counter()-total_started
    audit["unit_checkpointing"] = {"schema":1,"unit_count":len(unit_rows),"one_fresh_model_process_state_per_seed_batch_unit":True,"resume_uses_only_PASS_units_with_matching_task_and_input_config_sha256":True}
    atomic_json(out/f"{args.task}_calibration_audit.json",audit); print(json.dumps(audit,indent=2,sort_keys=True))


if __name__ == "__main__": main()
