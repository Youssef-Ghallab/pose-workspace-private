#!/usr/bin/env python3
"""Write a markdown comparison for LocSim runs with train-calibrated thresholds."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Write a markdown comparison for fixed-threshold LocSim runs.')
    parser.add_argument('--output-md', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-subset-ann', required=True)
    parser.add_argument('--oracle-tune-metrics', required=True)
    parser.add_argument('--oracle-val-metrics', required=True)
    parser.add_argument('--yolo-tune-metrics', required=True)
    parser.add_argument('--yolo-val-metrics', required=True)
    parser.add_argument('--yolo-weights', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--ratio', type=float, required=True)
    return parser.parse_args()


def load_metrics(path):
    with Path(path).open() as f:
        return json.load(f)['named_stats']


def main():
    args = parse_args()
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    oracle_tune = load_metrics(args.oracle_tune_metrics)
    oracle_val = load_metrics(args.oracle_val_metrics)
    yolo_tune = load_metrics(args.yolo_tune_metrics)
    yolo_val = load_metrics(args.yolo_val_metrics)

    lines = []
    lines.append('# Fixed-Threshold LocSim Comparison')
    lines.append('')
    lines.append(f'- Checkpoint: `{args.checkpoint}`')
    lines.append(f'- Threshold calibration split: `{args.ratio:.0%}` of training images')
    lines.append(f'- Train subset annotations: `{args.train_subset_ann}`')
    lines.append(f'- Calibration seed: `{args.seed}`')
    lines.append(f'- YOLO detector weights: `{args.yolo_weights}`')
    lines.append('')
    lines.append('## Calibrated Thresholds')
    lines.append('')
    lines.append('| Setting | Threshold Source | Fixed Threshold |')
    lines.append('| --- | --- | ---: |')
    lines.append(f"| Oracle | train 10% oracle boxes | {oracle_tune['score_threshold']:.6f} |")
    lines.append(f"| YOLO | train 10% YOLO boxes | {yolo_tune['score_threshold']:.6f} |")
    lines.append('')
    lines.append('## Validation Results')
    lines.append('')
    lines.append('| Setting | AP | AP50 | AP75 | Precision | Recall | F1 | ScoreTh | FrameAcc |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    lines.append(
        f"| Oracle | {oracle_val['AP']:.4f} | {oracle_val['AP_50']:.4f} | {oracle_val['AP_75']:.4f} | "
        f"{oracle_val['precision']:.4f} | {oracle_val['recall']:.4f} | {oracle_val['f1']:.4f} | "
        f"{oracle_val['score_threshold']:.6f} | {oracle_val['frame_accuracy']:.4f} |")
    lines.append(
        f"| YOLO | {yolo_val['AP']:.4f} | {yolo_val['AP_50']:.4f} | {yolo_val['AP_75']:.4f} | "
        f"{yolo_val['precision']:.4f} | {yolo_val['recall']:.4f} | {yolo_val['f1']:.4f} | "
        f"{yolo_val['score_threshold']:.6f} | {yolo_val['frame_accuracy']:.4f} |")
    lines.append('')
    lines.append('## Notes')
    lines.append('')
    lines.append('- These validation numbers use thresholds tuned on training data only, not on the validation split itself.')
    lines.append('- Oracle and YOLO use separate calibrated thresholds because their score distributions differ.')

    output_md.write_text('\n'.join(lines))
    print(f'Wrote {output_md}')


if __name__ == '__main__':
    main()
