# Upstream provenance and patch map

Experiments used `https://github.com/LeapLabTHU/ViTTT` at commit `e3477587d099e6b9e83e9e7c80b1b999e0989a20` and mmsegmentation at commit `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`.

Modified classification files:

- `vittt/models/ttt_block.py`: branch-local higher-order switches and fast-weight update control.
- `vittt/models/gradient_modes.py`: fixed modes and ordered block-by-branch path masks.

Modified segmentation file:

- `segmentation/mmseg/models/backbones/vittt.py`: equivalent path-mask integration, including the disclosed state-free scale compatibility fix.

Project experiment files:

- `gradpath_auto/calibrate.py`: calibration, gradient comparison, and saved-tensor cost measurement.
- `gradpath_auto/masks.py`: canonical mask validation/application.
- `gradpath_auto/mechanism.py`: mechanism checks.
- `benchmarks/training_kernel.py`: fixed-tensor resource characterization.

Files under `patches/` retain upstream attribution. Upstream licenses apply to upstream-derived portions; the repository MIT license covers original project code and documentation only.

