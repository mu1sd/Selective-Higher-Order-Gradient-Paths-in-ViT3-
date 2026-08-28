# Selective Higher-Order Gradient Paths in ViT3

Official reproduction-material repository for the manuscript:

> **Selective Higher-Order Gradient Paths for Efficient ViT3 Training: Performance, Memory, and Task-Dependent Limits**  
> Dianyuan Li, Xiamen University  
> Submitted to *The Journal of Supercomputing*

## Overview

This study investigates whether selected branch-local higher-order gradient
paths can be detached during ViT3 training while preserving the current
numerical forward values, model parameters, floating-point operations, and
inference graph. The experiments cover 34 candidate paths across 17 blocks and
compare Full-HO, Detach-DW, Detach-sGLU, First-Order, GradPath-Auto, and matched
controls.

The central result is task dependent. Selective path suppression can reduce
training cost within a prespecified performance tolerance, but the useful
branch differs between ImageNet-100 and LoveDA. First-Order is the fastest mode
in the controlled training-kernel benchmark; this repository and manuscript do
not claim that selective retention universally outperforms First-Order.

## Current public contents

The current public release contains:

- protocol and release-scope documentation;
- LoveDA post-hoc confirmatory-extension task contracts;
- frozen configurations and masks for that extension;
- formal result summaries and audit records for seeds 53 and 67;
- supporting tables and statistical outputs.

The repository does **not** currently contain the complete original GPU
campaign, restricted datasets, dataset images, source-initialization
checkpoints, final checkpoints, or private bulk logs. It must therefore be
treated as a public protocol/evidence release rather than a self-contained
end-to-end reproduction of every reported training run.

## Repository structure

```text
confirmatory_extension/
  analysis/       # confirmatory and combined statistical summaries
  audit/          # freeze, deployment, integrity, and evidence checks
  configs/        # LoveDA Full-HO, GradPath-Auto, and Random-K run configs
  formal/         # released formal-run records
  paper/          # paper-facing data and tables
  protocol/       # frozen confirmatory-extension protocol
  state/          # campaign state record
LICENSE_NOTICE.md
RELEASE_SCOPE.md
REPRODUCTION_GUIDE.md
```

## Important scientific labels

- Public method name: **GradPath-Auto**; immutable artifact label:
  `GSHPS_AUTO`.
- `RAND-SCHEDULE` and `RAND-PATH-K` are different controls.
- LoveDA seeds 53 and 67 are a post-hoc confirmatory extension and are not
  presented as preregistered evidence.
- Processed Cityscapes is supplementary-only and is not an official-resolution
  benchmark result.
- ADE20K continuations share their first 144,000 updates and are not
  independent from-scratch repeats.
- Random-K uses one static matched-cardinality mask per training seed; mask
  variance and training-seed variance are therefore not independently
  identified.

## Reproduction scope

Three levels should be distinguished:

1. **Public evidence inspection** -- inspect the released configurations,
   masks, result summaries, and audits in `confirmatory_extension/`.
2. **Protocol reproduction** -- independently obtain the datasets,
   initialization checkpoints, compatible upstream ViT3 implementation, and
   environment, then apply the released task contracts.
3. **Exact audit reproduction** -- additionally requires the immutable private
   campaign archives and omitted checkpoint artifacts retained by the author.

Exact numerical reproduction is not guaranteed without the same datasets,
initialization artifacts, software builds, seeds, and frozen task contracts.

## Recorded hardware and environment scope

The reported main resource measurements were collected in exclusive serial
runs on a single NVIDIA RTX 4090 (24 GB) host with:

- Ubuntu 22.04;
- 8 CPU cores;
- 16 GB system memory;
- Python 3.11;
- BF16 automatic mixed precision;
- PyTorch and the NVIDIA driver supplied by the execution image;
- no `torch.compile` in the controlled training-kernel benchmark.

The exact PyTorch, CUDA, cuDNN, and NVIDIA-driver build identifiers were not
preserved in the frozen records and are not reconstructed here. The reported
performance evidence should therefore be interpreted as a single-GPU
characterization, not a universal cross-platform claim.

## Data and checkpoints

No dataset or checkpoint is redistributed. Users must obtain ImageNet,
ADE20K, LoveDA, Cityscapes, and CIFAR-100 from their official providers and
comply with the corresponding licences and terms. Local dataset paths,
credentials, and private absolute paths must not be committed.

## Citation

If you use these materials, please cite the manuscript. Bibliographic metadata
and a DOI will be added after publication.

```bibtex
@article{li_selective_higher_order_vit3,
  title   = {Selective Higher-Order Gradient Paths for Efficient ViT3 Training:
             Performance, Memory, and Task-Dependent Limits},
  author  = {Li, Dianyuan},
  journal = {The Journal of Supercomputing},
  note    = {Manuscript submitted}
}
```

## Contact

**Dianyuan Li**  
College of Chemistry and Chemical Engineering, Xiamen University  
ORCID: [0009-0008-0809-3519](https://orcid.org/0009-0008-0809-3519)  
Email: ldy201216@163.com

## Licence and third-party components

Read `LICENSE_NOTICE.md` before reuse. The current repository does not grant an
open-source licence unless and until an explicit `LICENSE` file is added.
Third-party datasets, upstream ViT3 components, and mmsegmentation components
remain governed by their respective licences.
