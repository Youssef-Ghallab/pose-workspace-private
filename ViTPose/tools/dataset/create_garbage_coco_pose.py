#!/usr/bin/env python
"""Create a one-sample COCO-style keypoint dataset for smoke tests."""

from pathlib import Path
import json

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "garbage_coco"
ANN_ROOT = DATA_ROOT / "annotations"
IMG_ROOT = DATA_ROOT / "images"
IMAGE_NAME = "garbage_pose_000001.jpg"


def build_keypoints():
    points = [
        (96, 40),   # nose
        (84, 34),   # left_eye
        (108, 34),  # right_eye
        (72, 38),   # left_ear
        (120, 38),  # right_ear
        (68, 76),   # left_shoulder
        (124, 76),  # right_shoulder
        (58, 110),  # left_elbow
        (134, 110), # right_elbow
        (50, 144),  # left_wrist
        (142, 144), # right_wrist
        (76, 144),  # left_hip
        (116, 144), # right_hip
        (70, 186),  # left_knee
        (122, 186), # right_knee
        (66, 224),  # left_ankle
        (126, 224), # right_ankle
    ]
    keypoints = []
    for x, y in points:
        keypoints.extend([x, y, 2])
    return points, keypoints


def draw_image(points, path):
    image = Image.new("RGB", (192, 256), color=(245, 240, 232))
    draw = ImageDraw.Draw(image)

    draw.rectangle((24, 16, 168, 240), outline=(35, 85, 135), width=3)
    draw.ellipse((72, 18, 120, 66), outline=(90, 40, 25), width=3)
    draw.line((96, 66, 96, 144), fill=(90, 40, 25), width=5)
    draw.line((68, 76, 124, 76), fill=(90, 40, 25), width=5)
    draw.line((76, 144, 116, 144), fill=(90, 40, 25), width=5)

    skeleton = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6),
    ]
    for start, end in skeleton:
        draw.line((points[start][0], points[start][1], points[end][0], points[end][1]),
                  fill=(200, 80, 40), width=4)

    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(30, 120, 200))

    image.save(path, quality=95)


def build_annotation():
    _, keypoints = build_keypoints()
    return {
        "info": {
            "description": "One-sample garbage pose dataset for ViTPose smoke tests",
            "version": "1.0",
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": IMAGE_NAME,
                "width": 192,
                "height": 256,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [24, 16, 144, 224],
                "area": 32256,
                "iscrowd": 0,
                "num_keypoints": 17,
                "keypoints": keypoints,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow",
                    "right_elbow", "left_wrist", "right_wrist", "left_hip",
                    "right_hip", "left_knee", "right_knee", "left_ankle",
                    "right_ankle",
                ],
                "skeleton": [
                    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
                ],
            }
        ],
    }


def main():
    ANN_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_ROOT.mkdir(parents=True, exist_ok=True)

    points, _ = build_keypoints()
    draw_image(points, IMG_ROOT / IMAGE_NAME)
    annotation = build_annotation()

    for split in ("train", "val", "test"):
        ann_path = ANN_ROOT / f"person_keypoints_{split}2017.json"
        ann_path.write_text(json.dumps(annotation, indent=2))

    print(f"Created dummy dataset in {DATA_ROOT}")


if __name__ == "__main__":
    main()
