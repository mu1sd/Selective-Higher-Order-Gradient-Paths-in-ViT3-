# Basic reproduction guide

## What this package can reproduce

With a compatible Python/CUDA environment and independently obtained datasets,
this release provides the campaign source, dataset audits, frozen configuration
templates, training/evaluation entry points, analysis scripts, and the LoveDA
post-hoc confirmatory task contracts. It can therefore reproduce the **method
and protocol**, and can rebuild analyses from the included frozen evidence.

It cannot guarantee bit-identical recovery of the paper's released numbers
without the source initialization checkpoints and selected final checkpoints
that are deliberately omitted from GitHub.

## Setup

```bash
cd code/vit3_neurocomputing_r21_release
python -m venv .venv
# Activate .venv, then install the relevant dependency set.
pip install -r environment/requirements-classification.txt
```

For segmentation, follow `bootstrap_segmentation_environment.sh` and the
segmentation requirements. Do not use credentials or private absolute paths in
committed configuration files.

## Data and checkpoint contracts

Before formal execution, review and fill a *new local copy* of:

- `configs/rtx4090_campaign.example.json`
- `configs/full_campaign_master.example.json`
- `protocol/frozen_protocol.template.json`

Run the dataset audit/freeze steps documented in the code README. The release
requires ImageNet-100 and LoveDA roots, source initialization checkpoints, a
defined precision policy, a deadline, and a GPU-hour budget. These choices
must be frozen before reading new validation results.

## Entry points

The source release documents these principal commands:

```bash
python -m data.imagenet100 audit
python -m data.loveda audit
python -m campaign.generate_full_campaign --master /absolute/path/master.json --output /absolute/path/campaign.json
python -m training.classification train --config /absolute/path/run.json
python -m training.segmentation train --config /absolute/path/run.json
python -m analysis.noninferiority
```

Use the exact command syntax, argument set, and execution gates in
`code/vit3_neurocomputing_r21_release/README.md`; the abbreviated examples here
are navigational only.

## LoveDA post-hoc confirmation

`confirmatory_extension/` contains the frozen six-run task matrix, masks,
formal summaries, official evaluations, and the original/confirmatory/combined
statistics. Seeds 53 and 67 are post-hoc confirmatory evidence and must retain
that label in any downstream report.

## Verify before use

- Read `SOURCE_LOCK.json` and `RELEASE_MANIFEST.json`.
- Retain the distinction between `RAND-SCHEDULE` and `RAND-PATH-K`.
- Do not use processed Cityscapes for an official leaderboard comparison.
- Do not treat ADE20K shared-prefix continuations as independent repeats.
- Do not make inference-speed, FLOP, or parameter-reduction claims.
