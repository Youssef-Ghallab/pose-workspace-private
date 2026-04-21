#!/usr/bin/env python3
"""Generate COCO-style detector boxes with Ultralytics backends."""

import argparse
import gc
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate detector boxes with Ultralytics YOLO or RT-DETR.')
    parser.add_argument(
        '--detector-type',
        choices=['yolo', 'rtdetr'],
        required=True,
        help='Ultralytics detector backend.')
    parser.add_argument(
        '--detector-weights',
        required=True,
        help='Path to the detector weights file.')
    parser.add_argument(
        '--ann-file',
        required=True,
        help='COCO-style annotation file used to enumerate images.')
    parser.add_argument(
        '--img-prefix',
        required=True,
        help='Image directory prefix.')
    parser.add_argument(
        '--output-json',
        required=True,
        help='Where to write detector predictions in COCO bbox format.')
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-images', type=int, default=None)
    return parser.parse_args()


def chunked(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def load_model(detector_type, weights_path):
    if detector_type == 'yolo':
        from ultralytics import YOLO
        return YOLO(weights_path)
    from ultralytics import RTDETR
    return RTDETR(weights_path)


def main():
    args = parse_args()
    ann_file = Path(args.ann_file)
    img_prefix = Path(args.img_prefix)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with ann_file.open() as f:
        gt = json.load(f)

    images = gt['images']
    if args.max_images is not None:
        images = images[:args.max_images]
    image_paths = [img_prefix / image['file_name'] for image in images]
    image_id_by_name = {Path(img['file_name']).name: img['id'] for img in images}

    model = load_model(args.detector_type, args.detector_weights)

    detections = []
    det_id = 1
    total_chunks = max(1, int(math.ceil(len(image_paths) / float(args.batch_size))))

    for chunk_idx, image_chunk in enumerate(chunked(image_paths, args.batch_size), start=1):
        print(f'{args.detector_type} chunk {chunk_idx}/{total_chunks} ({len(image_chunk)} images)')
        results = model.predict(
            source=[str(path) for path in image_chunk],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            batch=min(args.batch_size, len(image_chunk)),
            save=False,
            verbose=False,
            stream=True)

        for image_path, result in zip(image_chunk, results):
            image_name = image_path.name
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

        # Ultralytics inference can hold onto sizeable CUDA buffers across many chunks.
        # Clearing them here keeps long validation runs stable on shared GPUs.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    output_json.write_text(json.dumps(detections, indent=2))
    print(f'Wrote {len(detections)} detections to {output_json}')


if __name__ == '__main__':
    main()
