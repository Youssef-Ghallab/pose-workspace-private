#!/usr/bin/env python3
"""Launcher for config-driven SoccerNet detector-to-pose evaluation."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run config-driven SoccerNet detector-to-pose evaluation.')
    parser.add_argument('--config', required=True, help='YAML config file.')
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_command(command):
    print('Running:')
    print(' '.join(command))
    subprocess.run(command, check=True)


def resolve_conda_executable():
    for candidate in (
            os.environ.get('CONDA_EXE'),
            shutil.which('conda'),
            '/home/youssef.ghallab/miniforge3/bin/conda'):
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError('Could not resolve a conda executable for cross-env evaluation')


def main():
    args = parse_args()
    cfg = load_config(args.config)

    repo_root = Path(cfg['repo_root'])
    pose_cfg = cfg['pose']
    detector_cfg = cfg['detector']
    data_cfg = cfg['data']
    eval_cfg = cfg['evaluation']
    run_name = cfg['run_name']
    conda_exe = resolve_conda_executable()

    output_dir = Path(eval_cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    if detector_cfg['type'] == 'oracle':
        command = [
            conda_exe, 'run', '-n', pose_cfg['conda_env'], '--no-capture-output',
            'python', '-u', str(repo_root / 'tools/analysis/eval_soccernet_oracle_locsim.py'),
            str(pose_cfg['config']),
            '--checkpoint', str(pose_cfg['checkpoint']),
            '--ann-file', str(data_cfg['ann_file']),
            '--img-prefix', str(data_cfg['img_prefix']),
            '--output-dir', str(output_dir),
            '--device', str(pose_cfg.get('device', 'cuda:0')),
            '--samples-per-gpu', str(pose_cfg.get('samples_per_gpu', 16)),
            '--workers-per-gpu', str(pose_cfg.get('workers_per_gpu', 4)),
        ]
        if eval_cfg.get('score_threshold') is not None:
            command.extend(['--score-threshold', str(eval_cfg['score_threshold'])])
        if eval_cfg.get('overwrite', True):
            command.append('--overwrite')
        run_command(command)
        return

    detector_json = output_dir / 'detector_predictions' / f'{run_name}__{detector_cfg["type"]}_boxes.json'

    detector_command = [
        conda_exe, 'run', '-n', detector_cfg['conda_env'], '--no-capture-output',
        'python', '-u', str(repo_root / 'tools/analysis/generate_ultralytics_detections.py'),
        '--detector-type', str(detector_cfg['type']),
        '--detector-weights', str(detector_cfg['weights']),
        '--ann-file', str(data_cfg['ann_file']),
        '--img-prefix', str(data_cfg['img_prefix']),
        '--output-json', str(detector_json),
        '--imgsz', str(detector_cfg.get('imgsz', 1024)),
        '--conf', str(detector_cfg.get('conf', 0.001)),
        '--iou', str(detector_cfg.get('iou', 0.7)),
        '--device', str(detector_cfg.get('device', 'cuda:0')),
        '--max-det', str(detector_cfg.get('max_det', 300)),
        '--batch-size', str(detector_cfg.get('batch_size', 16)),
    ]
    run_command(detector_command)

    pose_command = [
        conda_exe, 'run', '-n', pose_cfg['conda_env'], '--no-capture-output',
        'python', '-u', str(repo_root / 'tools/analysis/eval_pose_from_bbox_locsim.py'),
        str(pose_cfg['config']),
        '--checkpoint', str(pose_cfg['checkpoint']),
        '--bbox-file', str(detector_json),
        '--ann-file', str(data_cfg['ann_file']),
        '--img-prefix', str(data_cfg['img_prefix']),
        '--output-dir', str(output_dir),
        '--detector-label', str(detector_cfg.get('label', detector_cfg['type'].upper())),
        '--device', str(pose_cfg.get('device', 'cuda:0')),
        '--samples-per-gpu', str(pose_cfg.get('samples_per_gpu', 16)),
        '--workers-per-gpu', str(pose_cfg.get('workers_per_gpu', 4)),
        '--position-keypoint-index', str(eval_cfg.get('position_keypoint_index', 1)),
        '--threshold-tune-ratio', str(eval_cfg.get('threshold_tune_ratio', 0.0)),
        '--threshold-tune-seed', str(eval_cfg.get('threshold_tune_seed', 0)),
    ]

    if eval_cfg.get('score_threshold') is not None:
        pose_command.extend(['--score-threshold', str(eval_cfg['score_threshold'])])
    if eval_cfg.get('overwrite', True):
        pose_command.append('--overwrite')
    run_command(pose_command)


if __name__ == '__main__':
    main()
