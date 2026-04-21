#!/usr/bin/env python
"""Prepare Full HD SoccerNet annotations from 4K source annotations."""

import argparse
import json
from pathlib import Path
import zipfile


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract SoccerNet Full HD images and convert 4K annotations.'
    )
    parser.add_argument(
        '--ann-zip',
        default='/nfs-stor/youssef.ghallab/pose/data/SoccerNet/SpiideoSynLoc/4K/annotations.zip',
        help='Path to the 4K annotations zip archive.')
    parser.add_argument(
        '--images-root',
        default='/nfs-stor/youssef.ghallab/pose/data/SoccerNet/SpiideoSynLoc/fullhd',
        help='Root directory containing the Full HD train/val zip files.')
    parser.add_argument(
        '--output-root',
        default='data/soccernet_fhd',
        help='Output directory for converted annotations.')
    parser.add_argument(
        '--skip-extract',
        action='store_true',
        help='Skip extracting train.zip and val.zip if images are already unpacked.')
    return parser.parse_args()


def ensure_extracted(images_root, split):
    split_dir = images_root / split
    if split_dir.is_dir() and any(split_dir.glob('*.jpg')):
        print(f'Found extracted {split} images in {split_dir}')
        return

    zip_path = images_root / f'{split}.zip'
    if not zip_path.is_file():
        raise FileNotFoundError(f'Missing archive: {zip_path}')

    print(f'Extracting {zip_path} -> {images_root}')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(images_root)


def scale_point(point, sx, sy):
    return [round(point[0] * sx, 4), round(point[1] * sy, 4), int(point[2])]


def flatten_keypoints(points):
    flat = []
    for point in points:
        flat.extend(point)
    return flat


def convert_annotations(src_data, target_width, target_height):
    image_scale = {}
    converted_images = []

    for image in src_data['images']:
        sx = target_width / float(image['width'])
        sy = target_height / float(image['height'])
        image_scale[image['id']] = (sx, sy)
        converted_image = {
            'id': image['id'],
            'file_name': image['file_name'],
            'width': target_width,
            'height': target_height,
        }
        for optional_key in ('camera_matrix', 'undist_poly', 'dist_poly'):
            if optional_key in image:
                converted_image[optional_key] = image[optional_key]
        converted_images.append(converted_image)

    converted_annotations = []
    for ann in src_data['annotations']:
        sx, sy = image_scale[ann['image_id']]
        scaled_keypoints = [scale_point(point, sx, sy) for point in ann['keypoints']]
        flat_keypoints = flatten_keypoints(scaled_keypoints)
        bbox = ann['bbox']
        converted_annotations.append({
            'id': ann['id'],
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'bbox': [
                round(bbox[0] * sx, 4),
                round(bbox[1] * sy, 4),
                round(bbox[2] * sx, 4),
                round(bbox[3] * sy, 4),
            ],
            'area': round(float(ann['area']) * sx * sy, 4),
            'iscrowd': 0,
            'num_keypoints': sum(point[2] > 0 for point in scaled_keypoints),
            'keypoints': flat_keypoints,
            'keypoints_3d': ann.get('keypoints_3d'),
            'position_on_pitch': ann.get('position_on_pitch'),
        })

    categories = [{
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
        'keypoints': ['body_anchor', 'ground_contact'],
        'skeleton': [[1, 2]],
    }]

    return {
        'info': {
            'description': 'SoccerNet Full HD annotations converted from 4K source annotations',
            'version': '1.0',
        },
        'licenses': [],
        'images': converted_images,
        'annotations': converted_annotations,
        'categories': categories,
    }


def convert_split(ann_zip_path, ann_member, out_json):
    with zipfile.ZipFile(ann_zip_path) as zf, zf.open(ann_member) as f:
        src_data = json.load(f)

    converted = convert_annotations(
        src_data=src_data,
        target_width=1920,
        target_height=1080,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(converted))
    print(f'Wrote {out_json}')


def main():
    args = parse_args()
    ann_zip = Path(args.ann_zip)
    images_root = Path(args.images_root)
    output_root = Path(args.output_root)
    ann_out_root = output_root / 'annotations'

    if not args.skip_extract:
        ensure_extracted(images_root, 'train')
        ensure_extracted(images_root, 'val')

    convert_split(ann_zip, 'annotations/train.json', ann_out_root / 'train.json')
    convert_split(ann_zip, 'annotations/val.json', ann_out_root / 'val.json')


if __name__ == '__main__':
    main()
