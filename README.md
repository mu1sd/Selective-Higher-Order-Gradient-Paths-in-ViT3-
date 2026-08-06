# ViT3 Neurocomputing — Basic Reproduction Release

This repository is a compact, GitHub-oriented source release for the ViT3
Neurocomputing study. It contains the frozen r21 campaign implementation,
dataset adapters, GSHPS selection code, example configurations, analysis
utilities, and selected public evidence needed to inspect and rerun the study
protocol.

## Scope

This package supports **basic protocol reproduction** and rebuilding the
analysis from included frozen result extracts. It intentionally excludes:

- restricted datasets and dataset images;
- pretrained/source-initialization checkpoints and final checkpoints;
- private qualitative arrays, bulk logs, and redundant intermediate artifacts;
- Git history/cache files and generated build products.

Exact numerical reproduction of the released formal runs requires the same
datasets, source-initialization checkpoints, environment, seeds, and frozen
task contracts. The original immutable evidence archives remain the reference
for those artifacts.

## Quick start

```bash
python -m venv .venv
# activate the virtual environment, then select the applicable requirements file
pip install -r environment/requirements-classification.txt
# for segmentation tasks, also install the dependencies documented in
# bootstrap_segmentation_environment.sh / environment requirements
```

Read `SOURCE_LOCK.json`, `RELEASE_MANIFEST.json`, the relevant configuration
under `configs/`, and the campaign/GSHPS README before attempting a run.
The experiment code expects the user to configure local dataset and checkpoint
paths; no dataset or credential path is distributed in this repository.

## Important scientific labels

- Public method name: **GradPath-Auto**; artifact label: `GSHPS_AUTO`.
- `RAND-SCHEDULE` and `RAND-PATH-K` are different controls.
- LoveDA seeds 53 and 67 are post-hoc confirmatory extension evidence.
- Processed Cityscapes is supplementary-only.
- ADE20K shared-prefix continuations are not independent from-scratch repeats.

## Reproduction tiers

1. **Analysis/figure reproduction:** run the included analysis utilities on the
   frozen public evidence extracts.
2. **Protocol reproduction:** provide the documented public datasets and run
   the frozen configs on compatible hardware.
3. **Exact audit reproduction:** additionally obtain the selected checkpoints
   and immutable evidence archives retained by the author.

## Before public GitHub release

Read `LICENSE_NOTICE.md`. Do not make the repository public until you have
checked the upstream ViT3 and mmsegmentation licensing, data terms, and the
target journal's code/data policy.
