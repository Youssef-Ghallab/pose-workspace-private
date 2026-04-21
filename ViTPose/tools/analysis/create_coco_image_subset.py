#!/usr/bin/env python3
"""Create a random image-level subset of a COCO-style annotation file."""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a random image-level subset of a COCO annotation file.')
    parser.add_argument('--input-ann', required=True, help='Input COCO json.')
    parser.add_argument('--output-ann', required=True, help='Output subset json.')
    parser.add_argument(
        '--ratio',
        type=float,
        required=True,
        help='Fraction of images to keep, e.g. 0.1 for 10%%.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    return parser.parse_args()


def main():
    args = parse_args()
    input_ann = Path(args.input_ann)
    output_ann = Path(args.output_ann)

    with input_ann.open() as f:
        data = json.load(f)

    images = list(data['images'])
    if not images:
        raise ValueError(f'No images found in {input_ann}')

    keep_count = int(round(len(images) * args.ratio))
    keep_count = max(1, min(len(images), keep_count))

    image_ids = [img['id'] for img in images]
    rng = random.Random(args.seed)
    rng.shuffle(image_ids)
    keep_ids = set(image_ids[:keep_count])

    subset = {k: v for k, v in data.items() if k not in ('images', 'annotations')}
    subset['images'] = [img for img in images if img['id'] in keep_ids]
    subset['annotations'] = [
        ann for ann in data['annotations'] if ann['image_id'] in keep_ids
    ]

    output_ann.parent.mkdir(parents=True, exist_ok=True)
    output_ann.write_text(json.dumps(subset))

    print(f'Input images: {len(images)}')
    print(f'Kept images: {len(subset["images"])}')
    print(f'Kept annotations: {len(subset["annotations"])}')
    print(f'Wrote subset to {output_ann}')


if __name__ == '__main__':
    main()
