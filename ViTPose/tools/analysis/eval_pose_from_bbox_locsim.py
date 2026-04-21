#!/usr/bin/env python3
"""Evaluate a ViTPose checkpoint from an external bbox file using LocSim."""

import argparse
import copy
import gc
import json
import os
import random
import tempfile
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
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
        description='Evaluate a ViTPose checkpoint from detector bboxes.')
    parser.add_argument('config', help='Path to the ViTPose config.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--bbox-file', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--img-prefix', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--detector-label', default='Detector')
    parser.add_argument('--position-keypoint-index', type=int, default=1)
    parser.add_argument('--score-threshold', type=float, default=None)
    parser.add_argument('--threshold-tune-ratio', type=float, default=0.0)
    parser.add_argument('--threshold-tune-seed', type=int, default=0)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.089, 0.089])
    parser.add_argument('--samples-per-gpu', type=int, default=16)
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def subset_coco_annotations(src_ann_file: Path, max_images: int) -> Path:
    with src_ann_file.open() as f:
        data = json.load(f)

    image_ids = [img['id'] for img in data['images'][:max_images]]
    return subset_coco_annotations_by_ids(src_ann_file, image_ids)


def subset_coco_annotations_by_ids(src_ann_file: Path, image_ids) -> Path:
    image_ids_set = set(image_ids)
    with src_ann_file.open() as f:
        data = json.load(f)

    subset = {k: v for k, v in data.items() if k not in ('images', 'annotations')}
    subset['images'] = [img for img in data['images'] if img['id'] in image_ids_set]
    subset['annotations'] = [
        ann for ann in data['annotations'] if ann['image_id'] in image_ids_set
    ]

    handle, subset_path = tempfile.mkstemp(prefix='soccernet_eval_subset_', suffix='.json')
    os.close(handle)
    subset_path = Path(subset_path)
    subset_path.write_text(json.dumps(subset))
    return subset_path


def build_threshold_tune_split(src_ann_file: Path, ratio: float, seed: int):
    with src_ann_file.open() as f:
        data = json.load(f)

    image_ids = [img['id'] for img in data['images']]
    if not image_ids or ratio <= 0:
        return None, None

    tune_count = int(round(len(image_ids) * ratio))
    tune_count = max(1, min(len(image_ids) - 1, tune_count))

    shuffled = list(image_ids)
    random.Random(seed).shuffle(shuffled)
    tune_ids = shuffled[:tune_count]
    test_ids = shuffled[tune_count:]

    tune_ann = subset_coco_annotations_by_ids(src_ann_file, tune_ids)
    test_ann = subset_coco_annotations_by_ids(src_ann_file, test_ids)
    return tune_ann, test_ann


def build_cfg(config_path: Path, ann_file: Path, img_prefix: str, bbox_file: Path,
              samples_per_gpu: int, workers_per_gpu: int) -> Config:
    cfg = Config.fromfile(str(config_path))
    setup_multi_processes(cfg)
    cfg.model.pretrained = None

    cfg.data.samples_per_gpu = samples_per_gpu
    cfg.data.workers_per_gpu = workers_per_gpu
    cfg.data.test_dataloader = dict(samples_per_gpu=samples_per_gpu)

    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg.ann_file = str(ann_file)
    test_cfg.img_prefix = img_prefix
    test_cfg.data_cfg.use_gt_bbox = False
    test_cfg.data_cfg.det_bbox_thr = 0.0
    test_cfg.data_cfg.bbox_file = str(bbox_file)
    cfg.data.test = test_cfg
    return cfg


def build_bbox_lookup(dataset):
    lookup = {}
    for item in dataset.db:
        rel_path = os.path.relpath(item['image_file'], dataset.img_prefix)
        image_id = dataset.name2id[rel_path]
        lookup[(image_id, item['bbox_id'])] = item['bbox']
    return lookup


def pose_score_from_pred(pred_keypoints, box_score, vis_thr):
    scores = pred_keypoints[:, 2]
    keep = scores > vis_thr
    if keep.any():
        return float(scores[keep].mean() * box_score)
    return 0.0


def flatten_pred_keypoints(keypoints):
    return [float(v) for v in keypoints.reshape(-1)]


def predict_to_coco_detections(model, data_loader, dataset, bbox_lookup, device):
    model.eval()
    detections = []
    det_id = 1
    vis_thr = dataset.vis_thr
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    use_cuda = device.type == 'cuda' and torch.cuda.is_available()

    for batch in data_loader:
        imgs = batch['img'].to(device, non_blocking=use_cuda)
        img_metas = batch['img_metas'].data[0]
        with torch.no_grad():
            result = model(img=imgs, img_metas=img_metas, return_loss=False)

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

        del result
        for _ in range(imgs.shape[0]):
            prog_bar.update()
    return detections


def image_ids_from_ann_file(ann_file: Path):
    with ann_file.open() as f:
        data = json.load(f)
    return {img['id'] for img in data['images']}


def filter_predictions_by_image_ids(predictions, image_ids):
    return [pred for pred in predictions if pred['image_id'] in image_ids]


def evaluate_predictions(gt_path: Path, pred_json_path: Path, sigmas,
                         position_keypoint_index, score_threshold):
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


def write_markdown(markdown_path: Path, args, metrics_json_path: Path, metrics,
                   tuned_threshold=None):
    lines = []
    lines.append(f'# {args.detector_label} Detection LocSim Results')
    lines.append('')
    lines.append(f'- Checkpoint: `{args.checkpoint}`')
    lines.append(f'- Detector label: `{args.detector_label}`')
    lines.append(f'- BBox file: `{args.bbox_file}`')
    lines.append(f'- Ground-truth annotations: `{args.ann_file}`')
    lines.append(f'- Validation images: `{args.img_prefix}`')
    if tuned_threshold is not None and args.score_threshold is None and args.threshold_tune_ratio > 0:
        lines.append(f'- Fixed threshold from internal tune split: `{tuned_threshold:.6f}`')
    lines.append('')
    lines.append('| AP | AP50 | AP75 | Precision | Recall | F1 | ScoreTh | FrameAcc | Metrics JSON |')
    lines.append('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
    named = metrics['named_stats']
    lines.append(
        f"| {named['AP']:.4f} | {named['AP_50']:.4f} | {named['AP_75']:.4f} | "
        f"{named['precision']:.4f} | {named['recall']:.4f} | {named['f1']:.4f} | "
        f"{named['score_threshold']:.6f} | {named['frame_accuracy']:.4f} | "
        f"`{metrics_json_path.name}` |")
    lines.append('')
    markdown_path.write_text('\n'.join(lines))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    pose_dir = output_dir / 'pose_predictions'
    metrics_dir = output_dir / 'metrics'
    pose_dir.mkdir(parents=True, exist_ok=True)
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
        bbox_file=Path(args.bbox_file),
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu)

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
        drop_last=False)
    bbox_lookup = build_bbox_lookup(dataset)

    checkpoint_slug = Path(args.checkpoint).stem
    pose_pred_json_path = pose_dir / f'{checkpoint_slug}.json'
    metrics_json_path = metrics_dir / f'{checkpoint_slug}.json'
    tune_metrics_json_path = metrics_dir / f'{checkpoint_slug}__tune.json'
    tune_pose_pred_json_path = pose_dir / f'{checkpoint_slug}__tune.json'
    test_pose_pred_json_path = pose_dir / f'{checkpoint_slug}__test.json'
    tune_ann_file = None
    test_ann_file = None

    if pose_pred_json_path.exists() and metrics_json_path.exists() and not args.overwrite:
        metrics = json.loads(metrics_json_path.read_text())
        tuned_threshold = metrics['named_stats']['score_threshold']
    else:
        model = build_posenet(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        device = torch.device(args.device)
        model = model.to(device)

        pose_detections = predict_to_coco_detections(
            model=model,
            data_loader=data_loader,
            dataset=dataset,
            bbox_lookup=bbox_lookup,
            device=device)
        pose_pred_json_path.write_text(json.dumps(pose_detections, indent=2))

        if args.score_threshold is None and args.threshold_tune_ratio > 0:
            tune_ann_file, test_ann_file = build_threshold_tune_split(
                ann_for_eval, args.threshold_tune_ratio, args.threshold_tune_seed)
            tune_image_ids = image_ids_from_ann_file(tune_ann_file)
            test_image_ids = image_ids_from_ann_file(test_ann_file)

            tune_pose_detections = filter_predictions_by_image_ids(pose_detections, tune_image_ids)
            test_pose_detections = filter_predictions_by_image_ids(pose_detections, test_image_ids)
            tune_pose_pred_json_path.write_text(json.dumps(tune_pose_detections, indent=2))
            test_pose_pred_json_path.write_text(json.dumps(test_pose_detections, indent=2))

            tune_eval = evaluate_predictions(
                gt_path=tune_ann_file,
                pred_json_path=tune_pose_pred_json_path,
                sigmas=args.sigmas,
                position_keypoint_index=args.position_keypoint_index,
                score_threshold=None)
            tune_metrics = metrics_payload(tune_eval)
            tune_metrics_json_path.write_text(json.dumps(tune_metrics, indent=2))
            tuned_threshold = tune_metrics['named_stats']['score_threshold']

            test_eval = evaluate_predictions(
                gt_path=test_ann_file,
                pred_json_path=test_pose_pred_json_path,
                sigmas=args.sigmas,
                position_keypoint_index=args.position_keypoint_index,
                score_threshold=tuned_threshold)
            metrics = metrics_payload(test_eval)
        else:
            test_eval = evaluate_predictions(
                gt_path=ann_for_eval,
                pred_json_path=pose_pred_json_path,
                sigmas=args.sigmas,
                position_keypoint_index=args.position_keypoint_index,
                score_threshold=args.score_threshold)
            metrics = metrics_payload(test_eval)
            tuned_threshold = metrics['named_stats']['score_threshold']

        metrics_json_path.write_text(json.dumps(metrics, indent=2))

        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    markdown_path = output_dir / 'locsim_results.md'
    write_markdown(markdown_path, args, metrics_json_path, metrics, tuned_threshold)
    print(f'Saved markdown summary to {markdown_path}')

    if subset_ann_file is not None and subset_ann_file.exists():
        subset_ann_file.unlink()
    if tune_ann_file is not None and tune_ann_file.exists():
        tune_ann_file.unlink()
    if test_ann_file is not None and test_ann_file.exists():
        test_ann_file.unlink()


if __name__ == '__main__':
    main()
