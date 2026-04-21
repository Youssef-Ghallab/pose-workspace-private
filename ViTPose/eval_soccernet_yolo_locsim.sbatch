#!/bin/bash
#SBATCH --job-name=vitpose-yolo-locsim
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH --account=cscc-users
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --output=/nfs-stor/youssef.ghallab/pose/ViTPose/slurm_logs/vitpose_yolo_locsim_%j.log

set -euo pipefail

REPO_ROOT="/nfs-stor/youssef.ghallab/pose/ViTPose"
CONFIG_PATH="${REPO_ROOT}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_small_simple_soccernet_fhd_256x192.py"
CHECKPOINT_PATH="${REPO_ROOT}/work_dirs/soccernet_fhd/epoch_30.pth"
DETECTOR_WEIGHTS="${REPO_ROOT}/YOLO_weights/best.pt"
ANN_PATH="${REPO_ROOT}/data/soccernet_fhd/annotations/val.json"
IMG_PREFIX="/nfs-stor/youssef.ghallab/pose/data/SoccerNet/SpiideoSynLoc/fullhd/val/"
OUTPUT_DIR="${REPO_ROOT}/evaluation/yolo_detection"

cd "${REPO_ROOT}"
mkdir -p slurm_logs "${OUTPUT_DIR}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== ViTPose SoccerNet YOLO LocSim Evaluation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

conda run -n vitpose --no-capture-output python tools/dataset/prepare_soccernet_fullhd.py --skip-extract

conda run -n vitpose --no-capture-output python -u tools/analysis/eval_soccernet_yolo_locsim.py \
  "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --detector-weights "${DETECTOR_WEIGHTS}" \
  --ann-file "${ANN_PATH}" \
  --img-prefix "${IMG_PREFIX}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --samples-per-gpu 16 \
  --workers-per-gpu 4 \
  --detector-batch-size 16 \
  --overwrite
