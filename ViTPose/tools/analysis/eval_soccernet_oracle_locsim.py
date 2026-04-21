#!/usr/bin/env python3
"""Evaluate ViTPose SoccerNet checkpoints with oracle GT boxes using LocSim."""

import argparse
import copy
import gc
import json
import os
from collections import OrderedDict
from pathlib import Path
import tempfile

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from tqdm import tqdm
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes
from sskit.coco import LocSimCOCOeval

if not hasattr(torch.Tensor, 'mT'):
    torch.Tensor.mT = property(lambda self: self.transpose(-2, -1))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SoccerNet ViTPose checkpoints with oracle GT '
        'bounding boxes and LocSim.')
    parser.add_argument(
        'config',
        help='Path to the ViTPose config.')
    parser.add_argument(
        '--checkpoint',
        nargs='+',
        required=True,
        help='One or more checkpoint files to evaluate.')
    parser.add_argument(
        '--ann-file',
        default='/nfs-stor/youssef.ghallab/pose/ViTPose/data/soccernet_fhd/annotations/val.json',
        help='Validation annotation file with camera metadata.')
    parser.add_argument(
        '--img-prefix',
        default='/nfs-stor/youssef.ghallab/pose/data/SoccerNet/SpiideoSynLoc/fullhd/val/',
        help='Validation image directory.')
    parser.add_argument(
        '--output-dir',
        default='/nfs-stor/youssef.ghallab/pose/ViTPose/evaluation/oracle_detection',
        help='Directory for predictions, metrics, and markdown summary.')
    parser.add_argument(
        '--position-keypoint-index',
        type=int,
        default=1,
        help='Keypoint index used for pitch position projection. '
        'Default 1 = ground_contact.')
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=None,
        help='Optional fixed LocSim score threshold. Leave unset to let '
        'LocSimCOCOeval choose the F1-optimal validation threshold.')
    parser.add_argument(
        '--sigmas',
        type=float,
        nargs='+',
        default=[0.089, 0.089],
        help='Sigmas passed to LocSimCOCOeval.')
    parser.add_argument(
        '--samples-per-gpu',
        type=int,
        default=32,
        help='Batch size for evaluation.')
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        default=8,
        help='Number of dataloader workers.')
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Torch device to run inference on.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Optional smoke-test limit on number of images.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Recompute predictions/metrics even if output files exist.')
    return parser.parse_args()


def slugify_checkpoint_name(checkpoint_path: Path, repo_root: Path) -> str:
    rel = checkpoint_path.resolve().relative_to(repo_root.resolve())
    return str(rel).replace('/', '__').replace('.pth', '')


def subset_coco_annotations(src_ann_file: Path, max_images: int) -> Path:
    with src_ann_file.open() as f:
        data = json.load(f)

    image_ids = [img['id'] for img in data['images'][:max_images]]
    image_ids_set = set(image_ids)

    subset = {k: v for k, v in data.items() if k not in ('images', 'annotations')}
    subset['images'] = [img for img in data['images'] if img['id'] in image_ids_set]
    subset['annotations'] = [
        ann for ann in data['annotations'] if ann['image_id'] in image_ids_set
    ]

    handle, subset_path = tempfile.mkstemp(prefix='soccernet_val_subset_', suffix='.json')
    os.close(handle)
    subset_path = Path(subset_path)
    subset_path.write_text(json.dumps(subset))
    return subset_path


def build_cfg(config_path: Path, ann_file: Path, img_prefix: str, samples_per_gpu: int,
              workers_per_gpu: int) -> Config:
    cfg = Config.fromfile(str(config_path))
    setup_multi_processes(cfg)
    cfg.model.pretrained = None

    cfg.data.samples_per_gpu = samples_per_gpu
    cfg.data.workers_per_gpu = workers_per_gpu
    cfg.data.test_dataloader = dict(samples_per_gpu=samples_per_gpu)

    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg.ann_file = str(ann_file)
    test_cfg.img_prefix = img_prefix
    test_cfg.data_cfg.use_gt_bbox = True
    test_cfg.data_cfg.det_bbox_thr = 0.0
    test_cfg.data_cfg.bbox_file = ''
    cfg.data.test = test_cfg
    return cfg


def build_bbox_lookup(dataset):
    lookup = {}
    for item in dataset.db:
        rel_path = os.path.relpath(item['image_file'], dataset.img_prefix)
        image_id = dataset.name2id[rel_path]
        lookup[(image_id, item['bbox_id'])] = item['bbox']
    return lookup


def checkpoint_label(checkpoint_path: Path, repo_root: Path) -> str:
    return str(checkpoint_path.resolve().relative_to(repo_root.resolve()))


def flatten_pred_keypoints(keypoints):
    return [float(v) for v in keypoints.reshape(-1)]


def pose_score_from_pred(pred_keypoints, box_score, vis_thr):
    scores = pred_keypoints[:, 2]
    keep = scores > vis_thr
    if keep.any():
        return float(scores[keep].mean() * box_score)
    return 0.0


def predict_dataset(model, data_loader, device):
    model.eval()
    outputs = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for batch in data_loader:
        imgs = batch['img'].to(device, non_blocking=torch.cuda.is_available() and device.type == 'cuda')
        img_metas = batch['img_metas'].data[0]
        with torch.no_grad():
            result = model(img=imgs, img_metas=img_metas, return_loss=False)
        outputs.append(result)
        batch_size = imgs.shape[0]
        for _ in range(batch_size):
            prog_bar.update()
    return outputs


def outputs_to_coco_detections(outputs, dataset, bbox_lookup):
    detections = []
    det_id = 1
    vis_thr = dataset.vis_thr

    for result in outputs:
        preds = result['preds']
        boxes = result['boxes']
        image_paths = result['image_paths']
        bbox_ids = result['bbox_ids']

        for pred, box, image_path, bbox_id in zip(preds, boxes, image_paths, bbox_ids):
            rel_path = os.path.relpath(image_path, dataset.img_prefix)
            image_id = dataset.name2id[rel_path]
            bbox = bbox_lookup[(image_id, bbox_id)]
            score = pose_score_from_pred(pred, float(box[5]), vis_thr)
            detections.append({
                'id': det_id,
                'image_id': int(image_id),
                'category_id': 1,
                'bbox': [float(v) for v in bbox],
                'keypoints': flatten_pred_keypoints(pred),
                'score': score,
            })
            det_id += 1

    return detections


def metrics_payload(coco_eval):
    stats = [float(x) for x in coco_eval.stats]
    payload = {
        'stats': stats,
        'selected_metrics': {
            'mAP_LocSim': stats[0],
            'score_threshold': stats[15],
            'frame_accuracy': stats[16],
        },
    }
    if len(stats) >= 17:
        payload['named_stats'] = {
            'AP': stats[0],
            'AP_50': stats[1],
            'AP_75': stats[2],
            'AP_small': stats[3],
            'AP_medium': stats[4],
            'AP_large': stats[5],
            'AR_1': stats[6],
            'AR_10': stats[7],
            'AR_100': stats[8],
            'AR_small': stats[9],
            'AR_medium': stats[10],
            'AR_large': stats[11],
            'precision': stats[12],
            'recall': stats[13],
            'f1': stats[14],
            'score_threshold': stats[15],
            'frame_accuracy': stats[16],
        }
    return payload


def evaluate_predictions(gt_path: Path, pred_json_path: Path, sigmas, position_keypoint_index,
                         score_threshold):
    coco = COCO(str(gt_path))
    coco_det = coco.loadRes(str(pred_json_path))
    coco_eval = LocSimCOCOeval(coco, coco_det, 'bbox', sigmas, True)
    coco_eval.params.useSegm = None
    coco_eval.params.position_from_keypoint_index = position_keypoint_index
    if score_threshold is not None:
        coco_eval.params.score_threshold = score_threshold
    coco_eval.evaluate()
    coco_eval.accumulate()
    safe_summarize_locsim(coco_eval)
    return coco_eval


def safe_summarize_locsim(coco_eval):
    COCOeval.summarize(coco_eval)
    if hasattr(coco_eval.params, 'score_threshold'):
        threshold = coco_eval.params.score_threshold
    else:
        f1 = coco_eval.eval['f1_50']
        i = int(np.nanargmax(f1))
        scores = coco_eval.eval['scores_50']
        if i + 1 < len(scores):
            threshold = float((scores[i] + scores[i + 1]) / 2.0)
        else:
            threshold = float(scores[i])
    i = int(np.searchsorted(-coco_eval.eval['scores_50'], -threshold, 'right') - 1)
    i = max(0, min(i, len(coco_eval.eval['scores_50']) - 1))
    stats = [
        float(coco_eval.eval['precision_50'][i]),
        float(coco_eval.eval['recall_50'][i]),
        float(coco_eval.eval['f1_50'][i]),
        float(threshold),
        float(coco_eval.frame_accuracy(threshold)),
    ]
    coco_eval.stats = np.concatenate([coco_eval.stats, stats])

    print()
    print(f'  Precision      @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[0]:5.3f}')
    print(f'  Recall         @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[1]:5.3f}')
    print(f'  F1             @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[2]:5.3f}')
    print(f'  Frame Accuracy @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[4]:5.3f}')
    print(f'  mAP-LocSim     @[ LocSim=0.50:0.95 | ScoreTh={threshold:5.3f} ] = {float(coco_eval.stats[0]):5.3f}')


def write_markdown(rows, markdown_path: Path, ann_file: Path, img_prefix: str, max_images):
    lines = []
    lines.append('# Oracle Detection LocSim Results')
    lines.append('')
    lines.append(f'- Ground-truth boxes: `{ann_file}`')
    lines.append(f'- Validation images: `{img_prefix}`')
    lines.append('- Evaluation mode: oracle detection (GT bounding boxes as top-down inputs)')
    lines.append('- Position source: predicted keypoint index `1` (`ground_contact`)')
    if max_images is not None:
        lines.append(f'- Smoke-test subset: first `{max_images}` images only')
    lines.append('')
    lines.append('| Checkpoint | AP | AP50 | AP75 | Precision | Recall | F1 | ScoreTh | FrameAcc | Pred JSON | Metrics JSON |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |')

    for row in rows:
        lines.append(
            f"| `{row['checkpoint_label']}` | "
            f"{row['metrics']['named_stats']['AP']:.4f} | "
            f"{row['metrics']['named_stats']['AP_50']:.4f} | "
            f"{row['metrics']['named_stats']['AP_75']:.4f} | "
            f"{row['metrics']['named_stats']['precision']:.4f} | "
            f"{row['metrics']['named_stats']['recall']:.4f} | "
            f"{row['metrics']['named_stats']['f1']:.4f} | "
            f"{row['metrics']['named_stats']['score_threshold']:.6f} | "
            f"{row['metrics']['named_stats']['frame_accuracy']:.4f} | "
            f"`{row['pred_json'].name}` | `{row['metrics_json'].name}` |")
    lines.append('')

    if rows:
        best_row = max(rows, key=lambda item: item['metrics']['named_stats']['AP'])
        lines.append('## Best Checkpoint')
        lines.append('')
        lines.append(
            f"`{best_row['checkpoint_label']}` achieved the highest oracle-detection "
            f"mAP-LocSim at `{best_row['metrics']['named_stats']['AP']:.4f}` with "
            f"frame accuracy `{best_row['metrics']['named_stats']['frame_accuracy']:.4f}`.")
        lines.append('')

    markdown_path.write_text('\n'.join(lines))


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / 'predictions'
    metrics_dir = output_dir / 'metrics'
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ann_file = Path(args.ann_file)
    subset_ann_file = None
    if args.max_images is not None:
        subset_ann_file = subset_coco_annotations(ann_file, args.max_images)
        ann_for_eval = subset_ann_file
    else:
        ann_for_eval = ann_file

    cfg = build_cfg(
        config_path=Path(args.config),
        ann_file=ann_for_eval,
        img_prefix=args.img_prefix,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
    )

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
        drop_last=False,
    )
    bbox_lookup = build_bbox_lookup(dataset)

    device = torch.device(args.device)
    rows = []
    for checkpoint in args.checkpoint:
        checkpoint_path = Path(checkpoint)
        checkpoint_slug = slugify_checkpoint_name(checkpoint_path, repo_root)
        pred_json_path = pred_dir / f'{checkpoint_slug}.json'
        metrics_json_path = metrics_dir / f'{checkpoint_slug}.json'

        if pred_json_path.exists() and metrics_json_path.exists() and not args.overwrite:
            metrics = json.loads(metrics_json_path.read_text())
        else:
            print(f'\n=== Evaluating {checkpoint_path} ===')
            model = build_posenet(cfg.model)
            load_checkpoint(model, str(checkpoint_path), map_location='cpu')
            model = model.to(device)

            outputs = predict_dataset(model, data_loader, device)
            detections = outputs_to_coco_detections(outputs, dataset, bbox_lookup)
            pred_json_path.write_text(json.dumps(detections, indent=2))

            coco_eval = evaluate_predictions(
                gt_path=ann_for_eval,
                pred_json_path=pred_json_path,
                sigmas=args.sigmas,
                position_keypoint_index=args.position_keypoint_index,
                score_threshold=args.score_threshold,
            )
            metrics = metrics_payload(coco_eval)
            metrics_json_path.write_text(json.dumps(metrics, indent=2))

            del model
            del outputs
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        rows.append({
            'checkpoint_label': checkpoint_label(checkpoint_path, repo_root),
            'pred_json': pred_json_path,
            'metrics_json': metrics_json_path,
            'metrics': metrics,
        })

    markdown_path = output_dir / 'locsim_oracle_results.md'
    write_markdown(rows, markdown_path, ann_for_eval, args.img_prefix, args.max_images)
    print(f'\nSaved markdown summary to {markdown_path}')

    if subset_ann_file is not None and subset_ann_file.exists():
        subset_ann_file.unlink()


if __name__ == '__main__':
    main()
