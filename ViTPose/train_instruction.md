# ViTPose Training Instructions

This document explains how to run ViTPose training from scratch in this repo, including:

- cloning the repo
- creating a compatible conda environment
- preparing a one-sample dummy dataset for smoke testing
- running training on CPU or GPU
- starting and closing an interactive Slurm GPU session

## 1. Clone the Repository

```bash
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
```

## 2. Create the Conda Environment

Create the environment with Python 3.8:

```bash
source /home/youssef.ghallab/miniforge3/etc/profile.d/conda.sh
conda create -y -n vitpose python=3.8 pip
conda activate vitpose
```

Install common base packages:

```bash
pip install "numpy<2" cython
```

## 3. Install Dependencies

### CPU setup

If you only want CPU training:

```bash
pip install "torch==1.9.0+cpu" "torchvision==0.10.0+cpu" -f https://download.pytorch.org/whl/torch_stable.html
pip install "mmcv-full==1.3.9" -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html
```

### GPU setup

If you want GPU training, use the CUDA 11.1 wheels that work with this repo:

```bash
pip install "torch==1.9.0+cu111" "torchvision==0.10.0+cu111" -f https://download.pytorch.org/whl/torch_stable.html
pip install "mmcv-full==1.3.9" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

Then install the repo requirements and editable package:

```bash
pip install -r requirements.txt timm==0.4.9 einops
pip install "yapf==0.31.0"
pip install -v -e .
```

`yapf==0.31.0` is important here. Newer `yapf` versions break `mmcv==1.3.9` config pretty-printing.

Optional sanity check:

```bash
pip check
python - <<'PY'
import torch, mmcv, mmpose, timm
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("mmcv:", mmcv.__version__)
print("mmpose:", mmpose.__version__)
print("timm:", timm.__version__)
PY
```

## 4. Start a GPU Session with Slurm

If you are on the login node and need a GPU interactively, start a session like this:

```bash
srun --partition=long --qos=gpu-12 --gres=gpu:1 --mem=50G --cpus-per-task=16 --time=2:00:00 --pty bash
```

After the shell opens on the GPU node:

```bash
source /home/youssef.ghallab/miniforge3/etc/profile.d/conda.sh
conda activate vitpose
nvidia-smi
```

To verify that PyTorch can see the allocated GPU:

```bash
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
PY
```

When you are done, close the interactive session:

```bash
exit
```

## 5. Create the Dummy Smoke-Test Dataset

This repo includes a helper script that creates a one-image COCO-style keypoint dataset:

```bash
python tools/dataset/create_garbage_coco_pose.py
```

This generates:

- `data/garbage_coco/images/garbage_pose_000001.jpg`
- `data/garbage_coco/annotations/person_keypoints_train2017.json`
- `data/garbage_coco/annotations/person_keypoints_val2017.json`
- `data/garbage_coco/annotations/person_keypoints_test2017.json`

The smoke-test config is:

```bash
configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_simple_garbage_debug_256x192.py
```

## 6. Run Training

### GPU smoke test

```bash
export NO_ALBUMENTATIONS_UPDATE=1
python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_simple_garbage_debug_256x192.py \
  --work-dir work_dirs/garbage_smoke_gpu \
  --no-validate \
  --seed 0
```

### CPU smoke test

```bash
export NO_ALBUMENTATIONS_UPDATE=1
python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_simple_garbage_debug_256x192.py \
  --work-dir work_dirs/garbage_smoke \
  --no-validate \
  --seed 0
```

Expected behavior:

- the dataset loader reports `num_images: 1`
- training runs for `1` epoch
- a checkpoint named `epoch_1.pth` is written under the chosen `work_dir`

## 7. Check Outputs

Example output files:

```bash
find work_dirs/garbage_smoke_gpu -maxdepth 2 -type f | sort
```

Typical artifacts:

- `work_dirs/garbage_smoke_gpu/<timestamp>.log`
- `work_dirs/garbage_smoke_gpu/<timestamp>.log.json`
- `work_dirs/garbage_smoke_gpu/epoch_1.pth`

## 8. Train on a Real Dataset

For real training, replace the smoke-test config with one of the normal configs under:

```bash
configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/
```

You will need to:

- place your dataset under `data/...`
- update `ann_file` and `img_prefix`
- optionally point `model.pretrained` at a pretrained ViTPose checkpoint

Example:

```bash
python tools/train.py <your_config.py> --work-dir work_dirs/<run_name> --seed 0
```

## 9. Troubleshooting

If training crashes while printing the config, make sure:

```bash
pip install "yapf==0.31.0"
```

If `torch.cuda.is_available()` is `False` inside a GPU node, check:

- you are inside the `srun` shell
- you installed the `+cu111` PyTorch and torchvision wheels
- you installed the CUDA build of `mmcv-full`

If you want to recheck the environment:

```bash
pip check
```
