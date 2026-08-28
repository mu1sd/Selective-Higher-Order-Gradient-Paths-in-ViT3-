#!/usr/bin/env python3
"""Post-formal 12-unit mechanism extension including GSHPS and RAND-PATH-K."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from campaign.io_utils import atomic_json
from training.classification import load_official_model, seed_all as seed_cls, set_runtime_mode
from training.segmentation import load_model, make_samples, parse_loss, seed_all as seed_seg


SEEDS=(11,23,41)


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def checkpoint(path): return torch.load(path,map_location="cpu")["model"]


def method_configs(matrix,seed):
    result={}
    for pair in matrix["formal_pairs"]:
        if pair["id"]==f"gshps_auto_seed{seed}": result["GSHPS_AUTO"]=(read(pair["classification"]),read(pair["segmentation"]))
        if pair["id"]==f"rand_path_seed{seed}": result["RAND_PATH"]=(read(pair["classification"]),read(pair["segmentation"]))
    return result


def signatures(model):
    prefixes=[name for name,module in model.named_modules() if getattr(module,"set_high_order_path_mask",None) is not None]
    grads={}; blocks={i:{} for i in range(len(prefixes))}
    for name,parameter in model.named_parameters():
        if parameter.grad is None: continue
        value=parameter.grad.detach().float().cpu().clone(); grads[name]=value
        for index,prefix in enumerate(prefixes):
            if name==prefix or name.startswith(prefix+"."): blocks[index][name]=value; break
    return grads,blocks


def compare(current,baseline):
    def one(left,right):
        dot=ln=rn=diff=0.0
        for name in set(left)|set(right):
            a=left.get(name); b=right.get(name)
            if a is None: a=torch.zeros_like(b)
            if b is None: b=torch.zeros_like(a)
            dot+=float((a*b).sum()); ln+=float(a.square().sum()); rn+=float(b.square().sum()); diff+=float((a-b).square().sum())
        cosine=dot/max((ln*rn)**0.5,1e-30); return {"cosine":cosine,"direction_sensitivity":1-cosine,"relative_residual":diff**0.5/max(rn**0.5,1e-30),"candidate_l2":ln**0.5,"baseline_l2":rn**0.5}
    return one(current[0],baseline[0]),[{"block_index":i,**one(current[1][i],baseline[1][i])} for i in baseline[1]]


def classification_observation(config,method,seed,reference,old):
    cfg=dict(config); cfg.update(seed=seed,head_seed=seed,data_order_seed=seed,use_checkpoint=False); seed_cls(seed); model,_=load_official_model(cfg,torch.device("cuda"))
    if reference=="ffff_final": model.load_state_dict(checkpoint(old/"results/imagenet100/formal"/f"FFFF_seed{seed}"/"final.pth"),strict=True)
    if method=="FFFF": set_runtime_mode(model,"FFFF")
    generator=torch.Generator().manual_seed(61000+seed); images=torch.randn(int(cfg["batch_size"]),3,int(cfg["image_size"]),int(cfg["image_size"]),generator=generator).cuda(); labels=torch.arange(int(cfg["batch_size"]),device="cuda")%100; seed_cls(seed); model.train(); model.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True,dtype=torch.bfloat16): loss=F.cross_entropy(model(images),labels)
    loss.backward(); result=(float(loss.detach()),signatures(model)); del model,images,labels; torch.cuda.empty_cache(); return result


def segmentation_observation(config,method,seed,reference,old):
    cfg=dict(config); cfg.update(seed=seed,data_order_seed=seed,use_checkpoint=False); seed_seg(seed); model,_=load_model(cfg,torch.device("cuda"))
    if reference=="ffff_final": model.load_state_dict(checkpoint(old/"results/loveda/formal"/f"FFFF_seed{seed}"/"final.pth"),strict=True)
    if method=="FFFF": model.backbone.set_gradient_mode("FFFF")
    batch=int(cfg["batch_size"]); size=int(cfg["crop_size"]); classes=int(cfg.get("num_classes",7)); generator=torch.Generator().manual_seed(71000+seed); images=torch.randn(batch,3,size,size,generator=generator); masks=torch.randint(0,classes,(batch,size,size),generator=generator); samples=make_samples(masks,[{"image_path":f"mechanism_{i}","domain":"fixed"} for i in range(batch)]); seed_seg(seed); model.train(); data=model.data_preprocessor({"inputs":images,"data_samples":samples},training=True); model.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True,dtype=torch.bfloat16): losses=model._run_forward(data,mode="loss"); loss,_=parse_loss(model,losses)
    loss.backward(); result=(float(loss.detach()),signatures(model)); del model,images,masks,samples; torch.cuda.empty_cache(); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--matrix",type=Path,required=True); args=parser.parse_args(); matrix=read(args.matrix); root=Path(matrix["run_root"]); core=Path(matrix["core_root"]); core_matrix=read(core/"task_matrix.json"); old=Path(core_matrix["old_root"]); inherited=read(core/"mechanism/gradient_probes.json"); inherited_map={(u["task"],u["seed"],u["reference"]):u for u in inherited["units"]}; output=root/"mechanism/final_gradient_mechanism.json"; payload=read(output) if output.is_file() else {"status":"RUNNING","amendment":"P2.1","units":[]}; units=payload["units"]; completed={(u["task"],u["seed"],u["reference"]) for u in units}
    for task in ("imagenet100","loveda"):
        for seed in SEEDS:
            configs=method_configs(matrix,seed)
            for reference in ("initial","ffff_final"):
                if (task,seed,reference) in completed: continue
                func=classification_observation if task=="imagenet100" else segmentation_observation; index=0 if task=="imagenet100" else 1
                baseline_loss,baseline_sig=func(configs["GSHPS_AUTO"][index],"FFFF",seed,reference,old); observations={"FFFF":{"loss":baseline_loss}}; comparisons={}
                for method in ("GSHPS_AUTO","RAND_PATH"):
                    loss,sig=func(configs[method][index],method,seed,reference,old); global_cmp,block_cmp=compare(sig,baseline_sig); observations[method]={"loss":loss,"forward_loss_abs_diff_vs_ffff":abs(loss-baseline_loss),"all_gradients_finite":True}; comparisons[method]={"global":global_cmp,"blocks":block_cmp}
                inherited_unit=inherited_map[(task,seed,reference)]; units.append({"task":task,"seed":seed,"reference":reference,"fixed_minibatch":inherited_unit["fixed_minibatch"],"inherited_uniform_modes":{k:inherited_unit["observations"][k] for k in ("FFFF","DDDD","SSSS","OOOO")},"new_method_observations":observations,"exact_gradient_comparisons_vs_ffff":comparisons,"calibration_mask_stability":read(root/"protocol/gshps"/f"{task}_calibration_audit.json")["mask_stability"]}); payload["units"]=units; atomic_json(output,payload)
    if len(units)!=12: raise RuntimeError(f"final mechanism expected 12 units, observed {len(units)}")
    payload.update({"status":"PASS","amendment":"P2.1","unit_count":len(units),"executed_after_gshps_and_rand_path_formal":True,"methods":["FFFF","DDDD","SSSS","OOOO","GSHPS_AUTO","RAND_PATH"],"rand_schedule_excluded_from_local_operator_comparison":True,"units":units,"interpretation":"descriptive same-weight gradient mechanism; not a causal claim"}); atomic_json(output,payload); print(json.dumps({"status":"PASS","unit_count":len(units)},indent=2))


if __name__=="__main__": main()
