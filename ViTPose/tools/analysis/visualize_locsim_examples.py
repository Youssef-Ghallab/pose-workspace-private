#!/usr/bin/env python3
"""Visualize examples with different LocSim values on cropped player images."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from sskit.coco import image_to_ground


if not hasattr(torch.Tensor, 'mT'):
    torch.Tensor.mT = property(lambda self: self.transpose(-2, -1))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--pred-json', required=True)
    parser.add_argument('--img-prefix', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--position-keypoint-index', type=int, default=1)
    parser.add_argument(
        '--target-locsim',
        type=float,
        nargs='+',
        default=[0.95, 0.70, 0.40],
        help='Target LocSim values to visualize.')
    parser.add_argument('--crop-scale', type=float, default=2.0)
    parser.add_argument('--panel-width', type=int, default=220)
    parser.add_argument('--panel-height', type=int, default=220)
    parser.add_argument(
        '--exclude-images',
        nargs='*',
        default=[],
        help='Image file names to exclude when selecting examples.')
    return parser.parse_args()


def bbox_iou_xywh(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def keypoints_from_flat(flat):
    arr = np.array(flat, dtype=np.float32).reshape(-1, 3)
    return arr


def crop_with_context(image, bbox, scale):
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    crop_w = max(w * scale, 32.0)
    crop_h = max(h * scale, 32.0)
    x1 = max(0, int(math.floor(cx - crop_w / 2.0)))
    y1 = max(0, int(math.floor(cy - crop_h / 2.0)))
    x2 = min(image.shape[1], int(math.ceil(cx + crop_w / 2.0)))
    y2 = min(image.shape[0], int(math.ceil(cy + crop_h / 2.0)))
    return image[y1:y2, x1:x2].copy(), x1, y1


def draw_point(image, point_xy, origin_xy, color):
    x = int(round(point_xy[0] - origin_xy[0]))
    y = int(round(point_xy[1] - origin_xy[1]))
    cv2.circle(image, (x, y), 2, color, -1, lineType=cv2.LINE_AA)


def find_best_match(gt_ann, preds_for_image):
    best_pred = None
    best_iou = -1.0
    for pred in preds_for_image:
        iou = bbox_iou_xywh(gt_ann['bbox'], pred['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_pred = pred
    return best_pred, best_iou


def compute_locsim(image_info, gt_ann, pred_ann, position_keypoint_index):
    pred_kpts = keypoints_from_flat(pred_ann['keypoints'])
    pred_xy = pred_kpts[position_keypoint_index, :2]
    width = float(image_info['width'])
    height = float(image_info['height'])
    normalized = ((pred_xy - np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32))
                  / width).astype(np.float32)
    pred_bev = image_to_ground(
        image_info['camera_matrix'],
        image_info['undist_poly'],
        normalized.reshape(1, 2))[0, :2]
    if hasattr(pred_bev, 'detach'):
        pred_bev = pred_bev.detach().cpu().numpy()
    gt_bev = np.array(gt_ann['position_on_pitch'], dtype=np.float32)
    dist2 = float(np.sum((pred_bev - gt_bev) ** 2))
    locsim = float(np.exp(np.log(0.05) * dist2))
    return locsim, pred_bev, gt_bev


def resize_with_padding(image, target_width, target_height):
    src_h, src_w = image.shape[:2]
    scale = min(target_width / float(src_w), target_height / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    x0 = (target_width - new_w) // 2
    y0 = (target_height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def build_contact_sheet(originals, annotated, locsim_values, panel_width, panel_height):
    cols = len(originals)
    gap = 16
    caption_h = 34
    outer = 20
    rows = 2
    sheet_h = outer * 2 + rows * panel_height + rows * caption_h + (rows - 1) * gap
    sheet_w = outer * 2 + cols * panel_width + (cols - 1) * gap
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    def place_panel(image, row_idx, col_idx, locsim_value):
        x0 = outer + col_idx * (panel_width + gap)
        y0 = outer + row_idx * (panel_height + caption_h + gap)
        panel = resize_with_padding(image, panel_width, panel_height)
        sheet[y0:y0 + panel_height, x0:x0 + panel_width] = panel
        if row_idx == 1:
            caption = f'locsim = {locsim_value:.3f}'
            text_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
            text_x = x0 + max(0, (panel_width - text_size[0]) // 2)
            text_y = y0 + panel_height + 22
            cv2.putText(
                sheet,
                caption,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                1,
                lineType=cv2.LINE_AA)

    for idx, (orig, locsim_value) in enumerate(zip(originals, locsim_values)):
        place_panel(orig, 0, idx, locsim_value)
    for idx, (ann, locsim_value) in enumerate(zip(annotated, locsim_values)):
        place_panel(ann, 1, idx, locsim_value)
    return sheet


def main():
    args = parse_args()
    ann_path = Path(args.ann_file)
    pred_path = Path(args.pred_json)
    img_prefix = Path(args.img_prefix)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ann_path.open() as f:
        gt = json.load(f)
    with pred_path.open() as f:
        preds = json.load(f)
    excluded_images = set(args.exclude_images)

    images_by_id = {img['id']: img for img in gt['images']}
    preds_by_image = {}
    for pred in preds:
        preds_by_image.setdefault(pred['image_id'], []).append(pred)

    examples = []
    for gt_ann in gt['annotations']:
        image_id = gt_ann['image_id']
        image_info = images_by_id[image_id]
        if image_info['file_name'] in excluded_images:
            continue
        best_pred, best_iou = find_best_match(gt_ann, preds_by_image.get(image_id, []))
        if best_pred is None or best_iou <= 0:
            continue
        locsim, pred_bev, gt_bev = compute_locsim(
            image_info, gt_ann, best_pred, args.position_keypoint_index)
        pred_kpts = keypoints_from_flat(best_pred['keypoints'])
        gt_kpts = keypoints_from_flat(gt_ann['keypoints'])
        examples.append({
            'image_info': image_info,
            'gt_ann': gt_ann,
            'pred_ann': best_pred,
            'locsim': locsim,
            'best_iou': best_iou,
            'pred_kpts': pred_kpts,
            'gt_kpts': gt_kpts,
            'pred_bev': pred_bev,
            'gt_bev': gt_bev,
        })

    selected = []
    used = set()
    for target in args.target_locsim:
        ranked = sorted(
            ((idx, abs(example['locsim'] - target)) for idx, example in enumerate(examples)),
            key=lambda item: item[1])
        for idx, _ in ranked:
            if idx not in used:
                used.add(idx)
                selected.append(examples[idx])
                break

    summary_lines = [
        '# LocSim Visualization Examples',
        '',
        f'- Annotation file: `{ann_path}`',
        f'- Prediction file: `{pred_path}`',
        '',
        '| Example | Image | LocSim | Distance (m) | BBox IoU | File |',
        '| --- | --- | ---: | ---: | ---: | --- |',
    ]

    original_panels = []
    annotated_panels = []
    locsim_values = []

    for example_idx, example in enumerate(selected, start=1):
        image_info = example['image_info']
        image_path = img_prefix / image_info['file_name']
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        crop, x0, y0 = crop_with_context(image, example['gt_ann']['bbox'], args.crop_scale)
        original_crop = crop.copy()
        annotated_crop = crop.copy()
        for kp_idx in range(example['gt_kpts'].shape[0]):
            draw_point(annotated_crop, example['gt_kpts'][kp_idx, :2], (x0, y0), (0, 255, 0))
            draw_point(annotated_crop, example['pred_kpts'][kp_idx, :2], (x0, y0), (0, 0, 255))

        output_name = f'locsim_example_{example_idx:02d}_{example["locsim"]:.3f}.jpg'
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), annotated_crop)
        original_name = f'original_example_{example_idx:02d}_{example["locsim"]:.3f}.jpg'
        cv2.imwrite(str(output_dir / original_name), original_crop)
        distance_m = float(np.linalg.norm(example['pred_bev'] - example['gt_bev']))
        summary_lines.append(
            f'| {example_idx} | `{image_info["file_name"]}` | {example["locsim"]:.4f} | '
            f'{distance_m:.3f} | {example["best_iou"]:.3f} | `{output_name}` |')
        original_panels.append(original_crop)
        annotated_panels.append(annotated_crop)
        locsim_values.append(example['locsim'])

    if original_panels and annotated_panels:
        sheet = build_contact_sheet(
            originals=original_panels,
            annotated=annotated_panels,
            locsim_values=locsim_values,
            panel_width=args.panel_width,
            panel_height=args.panel_height)
        cv2.imwrite(str(output_dir / 'locsim_split_grid.jpg'), sheet)

    (output_dir / 'README.md').write_text('\n'.join(summary_lines) + '\n')


if __name__ == '__main__':
    main()
