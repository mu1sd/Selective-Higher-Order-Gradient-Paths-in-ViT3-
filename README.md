# Selective Higher-Order Gradient Paths in ViT3

Reproduction release for **Selective Higher-Order Gradient Paths for Efficient ViT3 Training: Performance, Memory, and Task-Dependent Limits**.

This repository exposes the method implementation, GradPath-Auto calibration, saved-tensor cost measurement, controlled training-kernel benchmark, frozen configurations, and selected result extracts. Datasets and checkpoints are not redistributed.

## Upstream lock

- Upstream: [LeapLabTHU/ViTTT](https://github.com/LeapLabTHU/ViTTT)
- ViT3 commit: `e3477587d099e6b9e83e9e7c80b1b999e0989a20`
- mmsegmentation commit: `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`
- Public method name: **GradPath-Auto**; frozen artifact key: `GSHPS_AUTO`

The modified classification and segmentation files are under `patches/`. See `UPSTREAM_AND_PATCHES.md` for the file map and attribution.

## Core implementation map

| Question | Location |
|---|---|
| 34 branch-local stop-gradient switches | `patches/classification/ttt_block.py`, `patches/segmentation/vittt.py` |
| Apply a 17-block x 2-branch mask | `patches/classification/gradient_modes.py` |
| GradPath-Auto calibration | `gradpath_auto/calibrate.py` |
| Saved-tensor byte measurement | `saved_bytes_context()` in `gradpath_auto/calibrate.py` |
| Mask validation and application | `gradpath_auto/masks.py` |
| Fixed-tensor resource benchmark | `benchmarks/training_kernel.py` |
| Frozen campaign examples | `configs/` |
| Repeated benchmark output | `results/resource_repeats.csv` |

## Environment

The formal measurements used one NVIDIA RTX 4090 (24 GB), 8 CPU cores, 16 GB host RAM, Ubuntu 22.04, Python 3.11, BF16 automatic mixed precision, and no `torch.compile`. Exact PyTorch, CUDA, cuDNN, and driver build identifiers were not preserved and are not guessed here.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/LeapLabTHU/ViTTT.git external/ViTTT
git -C external/ViTTT checkout e3477587d099e6b9e83e9e7c80b1b999e0989a20
```

Install the remaining upstream ViT3 dependencies. Copy `patches/classification/` into `external/ViTTT/vittt/models/`. For segmentation, copy `patches/segmentation/vittt.py` to the matching mmseg backbone location used by the frozen source tree. Keep a clean upstream checkout so the replacement can be audited with `git diff --no-index`.

## Calibration

Fill local dataset and checkpoint paths in the task configuration. Calibration consumes two frozen train-core minibatches for each of three checkpoint initializations, computes the Full-HO gradient, suppresses each candidate path in turn, and records direction, magnitude, and saved-tensor cost terms.

```bash
export PYTHONPATH="$PWD:$PWD/external/ViTTT"
python gradpath_auto/calibrate.py --help
```

## Controlled training-kernel benchmark

The benchmark uses task-shaped fixed synthetic tensors, 100 warm-up steps, 500 measured optimizer steps, CUDA events, `time.perf_counter`, and PyTorch peak allocated/reserved-memory APIs. Data loading and preprocessing are excluded.

```bash
export PYTHONPATH="$PWD:$PWD/external/ViTTT"
python benchmarks/training_kernel.py --matrix /path/to/filled_campaign_matrix.json
```

The frozen repeated measurements used in the paper are in `results/resource_repeats.csv`.

## Reproduction boundary

This release supports code inspection, method integration, calibration, controlled benchmarking, and rebuilding included summaries. Exact campaign reproduction additionally requires licensed datasets, source-initialization checkpoints, and filled local path contracts. No validation labels are used by GradPath-Auto. LoveDA seeds 53 and 67 are post-hoc confirmatory evidence; ADE20K continuations share a training prefix; processed Cityscapes is supplementary only.

## License

Original project code and documentation are released under the MIT License. Upstream ViT3 and mmsegmentation remain governed by their own licenses; dataset and checkpoint terms are unaffected. See `LICENSE` and `UPSTREAM_AND_PATCHES.md`.

