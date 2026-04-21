#!/bin/bash
#SBATCH --job-name=vitpose-oracle-locsim
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH --account=cscc-users
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --output=/nfs-stor/youssef.ghallab/pose/ViTPose/slurm_logs/vitpose_oracle_locsim_%j.log

set -euo pipefail

REPO_ROOT="/nfs-stor/youssef.ghallab/pose/ViTPose"
CONFIG_PATH="${REPO_ROOT}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_small_simple_soccernet_fhd_256x192.py"
ANN_PATH="${REPO_ROOT}/data/soccernet_fhd/annotations/val.json"
IMG_PREFIX="/nfs-stor/youssef.ghallab/pose/data/SoccerNet/SpiideoSynLoc/fullhd/val/"
OUTPUT_DIR="${REPO_ROOT}/evaluation/oracle_detection"

CHECKPOINTS=(
  "${REPO_ROOT}/work_dirs/soccernet_fhd/epoch_10.pth"
  "${REPO_ROOT}/work_dirs/soccernet_fhd/epoch_20.pth"
  "${REPO_ROOT}/work_dirs/soccernet_fhd/epoch_30.pth"
  "${REPO_ROOT}/work_dirs/soccernet_fhd/best_AP_epoch_30.pth"
  "${REPO_ROOT}/work_dirs/soccernet_fhd_nomosaic/epoch_10.pth"
  "${REPO_ROOT}/work_dirs/soccernet_fhd_nomosaic/epoch_20.pth"
  "${REPO_ROOT}/work_dirs/soccernet_fhd_nomosaic/epoch_30.pth"
  "${REPO_ROOT}/work_dirs/soccernet_fhd_nomosaic/epoch_40.pth"
  "${REPO_ROOT}/work_dirs/soccernet_fhd_nomosaic/epoch_50.pth"
  "${REPO_ROOT}/work_dirs/soccernet_fhd_nomosaic/best_AP_epoch_45.pth"
)

cd "${REPO_ROOT}"
mkdir -p slurm_logs "${OUTPUT_DIR}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== ViTPose SoccerNet Oracle LocSim Evaluation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

conda run -n vitpose --no-capture-output python tools/dataset/prepare_soccernet_fullhd.py --skip-extract

conda run -n vitpose --no-capture-output python -u tools/analysis/eval_soccernet_oracle_locsim.py \
  "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINTS[@]}" \
  --ann-file "${ANN_PATH}" \
  --img-prefix "${IMG_PREFIX}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --samples-per-gpu 32 \
  --workers-per-gpu 8 \
  --overwrite
