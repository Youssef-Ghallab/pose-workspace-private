# yolo26det

This directory contains small Ultralytics-based scripts used for detector experiments in the SoccerNet / SpiideoSynLoc workflow.

It is not the main project entrypoint. For the full pose-and-localization pipeline, start from:

- [`../README.md`](../README.md)
- [`../ViTPose/README_PROJECT.md`](../ViTPose/README_PROJECT.md)

## Files

- [`train.py`](./train.py): a compact Ultralytics training launcher.
- [`evaluate_spiideo_val.py`](./evaluate_spiideo_val.py): validation-time detector + LocSim evaluation script.

## What This Folder Is For

Use this directory if you want to:

- inspect the standalone detector training command
- inspect the earlier validation evaluation script
- compare the detector-side baseline against the integrated ViTPose evaluation flow

## What To Use For Main Experiments

The maintained experiment flow in this workspace is under [`../ViTPose/`](../ViTPose), especially:

- [`../ViTPose/README_PROJECT.md`](../ViTPose/README_PROJECT.md)
- [`../ViTPose/tools/analysis/eval_soccernet_from_config.py`](../ViTPose/tools/analysis/eval_soccernet_from_config.py)
- [`../ViTPose/configs/eval/`](../ViTPose/configs/eval)

That path is the one used for:

- oracle-box evaluation
- YOLO-box evaluation
- RT-DETR-box evaluation
- fixed-threshold validation/test runs

## Caution

These scripts contain hard-coded paths and training settings from the copied source environment. Before reusing them directly, check:

- model checkpoint paths
- dataset paths
- output locations
- W\&B login / experiment naming

For reproducible runs inside this workspace, prefer the config-driven and Slurm-based flow in [`../ViTPose/`](../ViTPose).
