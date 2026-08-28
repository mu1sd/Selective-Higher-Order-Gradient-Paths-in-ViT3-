#!/usr/bin/env python3
"""Exclusive-GPU task-shaped profiles for the two new path-mask methods."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from campaign.io_utils import atomic_json
from training.classification import load_official_model, seed_all as seed_cls
from training.segmentation import load_model, make_samples, optimizer_for, parse_loss, seed_all as seed_seg


def percentile(values,q): return sorted(values)[int(q*(len(values)-1))]


def payload(task,method,repeat,batch,e2e,gpu,device,config):
    mask=json.loads(Path(config["path_mask_path"]).read_text(encoding="utf-8"))
    return {"task":task,"method":method,"repeat":repeat,"warmup_steps":100,"measured_steps":500,"batch_size":batch,"e2e_p50_seconds":statistics.median(e2e),"e2e_p95_seconds":percentile(e2e,.95),"gpu_p50_seconds":statistics.median(gpu),"gpu_p95_seconds":percentile(gpu,.95),"samples_per_second_e2e":batch/statistics.median(e2e),"samples_per_second_gpu":batch/statistics.median(gpu),"peak_allocated_mb":torch.cuda.max_memory_allocated(device)/1024**2,"peak_reserved_mb":torch.cuda.max_memory_reserved(device)/1024**2,"precision":"bf16","kept_paths":mask["kept_paths"],"total_paths":mask["total_paths"],"kept_path_ratio":mask["kept_paths"]/mask["total_paths"],"scheduler_context":"serial_exclusive_gpu","resource_scope":"task-shaped fixed synthetic tensors; training kernel only"}


def classification_once(base,method,repeat):
    config=dict(base); config["seed"]=20260720+repeat; config["head_seed"]=20260720; device=torch.device("cuda"); seed_cls(config["seed"]); model,_=load_official_model(config,device); model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["learning_rate"]),weight_decay=float(config["weight_decay"])); generator=torch.Generator().manual_seed(82000+repeat); images=torch.randn(int(config["batch_size"]),3,int(config["image_size"]),int(config["image_size"]),generator=generator).cuda(); labels=torch.arange(int(config["batch_size"]),device=device)%100; e2e=[]; gpu=[]; torch.cuda.reset_peak_memory_stats(device)
    for step in range(600):
        started=time.perf_counter(); begin=torch.cuda.Event(True); end=torch.cuda.Event(True); begin.record(); optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True,dtype=torch.bfloat16): loss=F.cross_entropy(model(images),labels)
        loss.backward(); optimizer.step(); end.record(); torch.cuda.synchronize(device)
        if step>=100: gpu.append(begin.elapsed_time(end)/1000); e2e.append(time.perf_counter()-started)
    result=payload("imagenet100",method,repeat,int(config["batch_size"]),e2e,gpu,device,config); del model,optimizer,images,labels; torch.cuda.empty_cache(); return result


def segmentation_once(base,method,repeat):
    config=dict(base); config["seed"]=20260720+repeat; device=torch.device("cuda"); seed_seg(config["seed"]); model,_=load_model(config,device); _,wrapper=optimizer_for(model,config); batch=int(config["batch_size"]); size=int(config["crop_size"]); classes=int(config.get("num_classes",7)); generator=torch.Generator().manual_seed(92000+repeat); images=torch.randn(batch,3,size,size,generator=generator); masks=torch.randint(0,classes,(batch,size,size),generator=generator); samples=make_samples(masks,[{"image_path":f"synthetic_{i}","domain":"synthetic"} for i in range(batch)]); e2e=[]; gpu=[]; torch.cuda.reset_peak_memory_stats(device)
    for step in range(600):
        started=time.perf_counter(); begin=torch.cuda.Event(True); end=torch.cuda.Event(True); begin.record(); model.train(); data=model.data_preprocessor({"inputs":images,"data_samples":samples},training=True)
        with wrapper.optim_context(model):
            with torch.cuda.amp.autocast(enabled=True,dtype=torch.bfloat16): losses=model._run_forward(data,mode="loss"); loss,_=parse_loss(model,losses)
        wrapper.update_params(loss); end.record(); torch.cuda.synchronize(device)
        if step>=100: gpu.append(begin.elapsed_time(end)/1000); e2e.append(time.perf_counter()-started)
    result=payload("loveda",method,repeat,batch,e2e,gpu,device,config); del model,wrapper,images,masks,samples; torch.cuda.empty_cache(); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--matrix",type=Path,required=True); args=parser.parse_args(); matrix=json.loads(args.matrix.read_text(encoding="utf-8")); root=Path(matrix["run_root"]); output=root/"resource"/"gshps_serial_profiles.json"; data=json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {"status":"RUNNING","measurements":[]}; done={(x["task"],x["method"],x["repeat"]) for x in data["measurements"]}
    configs={}
    for pair in matrix["formal_pairs"]:
        if pair["id"] in ("gshps_auto_seed11","rand_path_seed11","gshps_sens_seed11"):
            method="GSHPS_AUTO" if pair["id"].startswith("gshps_auto") else ("GSHPS_SENS" if pair["id"].startswith("gshps_sens") else "RAND_PATH"); configs[("imagenet100",method)]=json.loads(Path(pair["classification"]).read_text(encoding="utf-8")); configs[("loveda",method)]=json.loads(Path(pair["segmentation"]).read_text(encoding="utf-8"))
    for task,func in (("imagenet100",classification_once),("loveda",segmentation_once)):
        for method in ("GSHPS_AUTO","RAND_PATH","GSHPS_SENS"):
            for repeat in range(3):
                if (task,method,repeat) in done: continue
                data["measurements"].append(func(configs[(task,method)],method,repeat)); atomic_json(output,data)
    data.update({"status":"PASS","measurement_count":18,"exclusive_gpu_required":True}); atomic_json(output,data); print(json.dumps(data,indent=2,sort_keys=True))


if __name__=="__main__": main()
