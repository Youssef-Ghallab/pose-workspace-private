#!/usr/bin/env python3
"""Evaluate a ViTPose SoccerNet checkpoint using YOLO detections and LocSim."""

import argparse
import copy
import gc
import json
import os
import random
from pathlib import Path
import tempfile
from typing import List

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
from ultralytics import YOLO

if not hasattr(torch.Tensor, 'mT'):
    torch.Tensor.mT = property(lambda self: self.transpose(-2, -1))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a SoccerNet ViTPose checkpoint with YOLO '
        'detections and LocSim.')
    parser.add_argument('config', help='Path to the ViTPose config.')
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Checkpoint file to evaluate.')
    parser.add_argument(
        '--detector-weights',
        required=True,
        help='YOLO detector weights used to generate input bounding boxes.')
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
        default='/nfs-stor/youssef.ghallab/pose/ViTPose/evaluation/yolo_detection',
        help='Directory for detections, pose predictions, metrics, and summary.')
    parser.add_argument(
        '--position-keypoint-index',
        type=int,
        default=1,
        help='Keypoint index used for pitch position projection.')
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=None,
        help='Optional fixed LocSim score threshold.')
    parser.add_argument(
        '--threshold-tune-ratio',
        type=float,
        default=0.1,
        help='Fraction of evaluation images reserved to tune the LocSim '
        'score threshold. Set to 0 to disable split tuning.')
    parser.add_argument(
        '--threshold-tune-seed',
        type=int,
        default=0,
        help='Random seed for the threshold tuning split.')
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
        help='Batch size for pose evaluation.')
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        default=8,
        help='Number of pose dataloader workers.')
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Torch device to run both detector and pose model on.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Optional smoke-test limit on number of images.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Recompute detector boxes, pose predictions, and metrics.')
    parser.add_argument(
        '--imgsz',
        type=int,
        default=1024,
        help='YOLO inference image size.')
    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='YOLO detection confidence threshold.')
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='YOLO NMS IoU threshold.')
    parser.add_argument(
        '--max-det',
        type=int,
        default=300,
        help='Maximum detections per image for YOLO.')
    parser.add_argument(
        '--detector-batch-size',
        type=int,
        default=64,
        help='Number of images to hand to YOLO per prediction chunk.')
    return parser.parse_args()


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


def subset_coco_annotations_by_ids(src_ann_file: Path, image_ids) -> Path:
    image_ids_set = set(image_ids)
    with src_ann_file.open() as f:
        data = json.load(f)

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


def chunked(items: List[Path], chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def build_detector_detections(model, images, image_id_by_name, imgsz, conf, iou, device,
                              max_det, batch_size):
    detections = []
    det_id = 1
    total_chunks = max(1, int(np.ceil(len(images) / float(batch_size))))

    for chunk_idx, image_chunk in enumerate(chunked(images, batch_size), start=1):
        print(f'YOLO chunk {chunk_idx}/{total_chunks} ({len(image_chunk)} images)')
        results = model.predict(
            source=[str(path) for path in image_chunk],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            batch=min(batch_size, len(image_chunk)),
            save=False,
            verbose=False,
            stream=True)

        for image_path, result in zip(image_chunk, results):
            image_name = Path(image_path).name
            image_id = image_id_by_name.get(image_name)
            if image_id is None or result.boxes is None:
                continue

            boxes_xyxy = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            for box_xyxy, score in zip(boxes_xyxy, scores):
                x1, y1, x2, y2 = box_xyxy
                detections.append({
                    'id': det_id,
                    'image_id': int(image_id),
                    'category_id': 1,
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score': float(score),
                })
                det_id += 1

        gc.collect()
        if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return detections


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

    print()
    print(f'  Precision      @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[0]:5.3f}')
    print(f'  Recall         @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[1]:5.3f}')
    print(f'  F1             @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[2]:5.3f}')
    print(f'  Frame Accuracy @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[4]:5.3f}')
    print(f'  mAP-LocSim     @[ LocSim=0.50:0.95 | ScoreTh={threshold:5.3f} ] = {float(coco_eval.stats[0]):5.3f}')


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


def image_ids_from_ann_file(ann_file: Path):
    with ann_file.open() as f:
        data = json.load(f)
    return {img['id'] for img in data['images']}


def filter_predictions_by_image_ids(predictions, image_ids):
    return [pred for pred in predictions if pred['image_id'] in image_ids]


def write_markdown(markdown_path: Path, args, detector_json_path: Path,
                   pose_pred_json_path: Path, metrics_json_path: Path, metrics,
                   ann_file: Path, tuned_threshold=None, tune_ann_file=None,
                   tune_metrics_json_path=None):
    lines = []
    lines.append('# YOLO Detection LocSim Results')
    lines.append('')
    lines.append(f'- Checkpoint: `{args.checkpoint}`')
    lines.append(f'- Detector weights: `{args.detector_weights}`')
    lines.append(f'- Ground-truth annotations: `{ann_file}`')
    lines.append(f'- Validation images: `{args.img_prefix}`')
    lines.append('- Evaluation mode: detector boxes from YOLO, then top-down pose on those boxes')
    lines.append('- Position source: predicted keypoint index `1` (`ground_contact`)')
    lines.append(
        f'- YOLO inference: `imgsz={args.imgsz}`, `conf={args.conf}`, '
        f'`iou={args.iou}`, `max_det={args.max_det}`, '
        f'`detector_batch_size={args.detector_batch_size}`')
    if tune_ann_file is not None:
        lines.append(
            f'- Threshold tuning split: `{args.threshold_tune_ratio:.0%}` '
            f'tune / `{1.0 - args.threshold_tune_ratio:.0%}` test '
            f'(seed `{args.threshold_tune_seed}`)')
        lines.append(f'- Tune annotations: `{tune_ann_file}`')
        lines.append(f'- Fixed test threshold from tune split: `{tuned_threshold:.6f}`')
        lines.append(f'- Tune metrics JSON: `{tune_metrics_json_path.name}`')
    if args.max_images is not None:
        lines.append(f'- Smoke-test subset: first `{args.max_images}` images only')
    lines.append('')
    lines.append('| AP | AP50 | AP75 | Precision | Recall | F1 | ScoreTh | FrameAcc | Detector JSON | Pose JSON | Metrics JSON |')
    lines.append('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |')
    named = metrics['named_stats']
    lines.append(
        f"| {named['AP']:.4f} | {named['AP_50']:.4f} | {named['AP_75']:.4f} | "
        f"{named['precision']:.4f} | {named['recall']:.4f} | {named['f1']:.4f} | "
        f"{named['score_threshold']:.6f} | {named['frame_accuracy']:.4f} | "
        f"`{detector_json_path.name}` | `{pose_pred_json_path.name}` | `{metrics_json_path.name}` |")
    lines.append('')
    markdown_path.write_text('\n'.join(lines))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    det_dir = output_dir / 'detector_predictions'
    pose_dir = output_dir / 'pose_predictions'
    metrics_dir = output_dir / 'metrics'
    det_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ann_file = Path(args.ann_file)
    subset_ann_file = None
    if args.max_images is not None:
        subset_ann_file = subset_coco_annotations(ann_file, args.max_images)
        ann_for_eval = subset_ann_file
    else:
        ann_for_eval = ann_file

    with ann_for_eval.open() as f:
        gt = json.load(f)
    images = [Path(args.img_prefix) / img['file_name'] for img in gt['images']]
    image_id_by_name = {Path(img['file_name']).name: img['id'] for img in gt['images']}

    checkpoint_slug = Path(args.checkpoint).stem
    det_slug = f'{checkpoint_slug}__yolo_boxes'
    detector_json_path = det_dir / f'{det_slug}.json'
    pose_pred_json_path = pose_dir / f'{checkpoint_slug}.json'
    metrics_json_path = metrics_dir / f'{checkpoint_slug}.json'
    tune_metrics_json_path = metrics_dir / f'{checkpoint_slug}__tune.json'
    tune_pose_pred_json_path = pose_dir / f'{checkpoint_slug}__tune.json'
    test_pose_pred_json_path = pose_dir / f'{checkpoint_slug}__test.json'
    tune_ann_file = None
    test_ann_file = None

    if not detector_json_path.exists() or args.overwrite:
        print(f'\n=== Running YOLO detections with {args.detector_weights} ===')
        detector = YOLO(args.detector_weights)
        detections = build_detector_detections(
            model=detector,
            images=images,
            image_id_by_name=image_id_by_name,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            batch_size=args.detector_batch_size)
        detector_json_path.write_text(json.dumps(detections, indent=2))
        del detector
        gc.collect()
        if args.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

    cfg = build_cfg(
        config_path=Path(args.config),
        ann_file=ann_for_eval,
        img_prefix=args.img_prefix,
        bbox_file=detector_json_path,
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

    if pose_pred_json_path.exists() and metrics_json_path.exists() and not args.overwrite:
        metrics = json.loads(metrics_json_path.read_text())
        tuned_threshold = metrics['named_stats']['score_threshold']
    else:
        print(f'\n=== Evaluating pose checkpoint {args.checkpoint} ===')
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

            tune_pose_detections = filter_predictions_by_image_ids(
                pose_detections, tune_image_ids)
            test_pose_detections = filter_predictions_by_image_ids(
                pose_detections, test_image_ids)
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
            tuned_threshold = args.score_threshold
            coco_eval = evaluate_predictions(
                gt_path=ann_for_eval,
                pred_json_path=pose_pred_json_path,
                sigmas=args.sigmas,
                position_keypoint_index=args.position_keypoint_index,
                score_threshold=args.score_threshold)
            metrics = metrics_payload(coco_eval)
            tuned_threshold = metrics['named_stats']['score_threshold']
        metrics_json_path.write_text(json.dumps(metrics, indent=2))

        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    markdown_path = output_dir / 'locsim_yolo_results.md'
    write_markdown(
        markdown_path=markdown_path,
        args=args,
        detector_json_path=detector_json_path,
        pose_pred_json_path=pose_pred_json_path,
        metrics_json_path=metrics_json_path,
        metrics=metrics,
        ann_file=ann_for_eval,
        tuned_threshold=tuned_threshold,
        tune_ann_file=tune_ann_file if args.score_threshold is None and args.threshold_tune_ratio > 0 else None,
        tune_metrics_json_path=tune_metrics_json_path if args.score_threshold is None and args.threshold_tune_ratio > 0 else None)
    print(f'\nSaved markdown summary to {markdown_path}')

    if subset_ann_file is not None and subset_ann_file.exists():
        subset_ann_file.unlink()
    if args.score_threshold is None and args.threshold_tune_ratio > 0:
        if tune_ann_file is not None and tune_ann_file.exists():
            tune_ann_file.unlink()
        if test_ann_file is not None and test_ann_file.exists():
            test_ann_file.unlink()


if __name__ == '__main__':
    main()
