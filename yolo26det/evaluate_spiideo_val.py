#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
from xtcocotools.coco import COCO
from sskit.coco import LocSimCOCOeval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an Ultralytics pose model on SpiideoSynLoc val with sskit LocSim."
    )
    parser.add_argument(
        "--model",
        default="/home/mohamed.abouelhadid/cv703/runs/pose/cv703/baseline/weights/best.pt",
        help="Path to the Ultralytics pose checkpoint.",
    )
    parser.add_argument(
        "--gt",
        default="/l/users/mohamed.abouelhadid/SpiideoSynLoc/annotations/val.json",
        help="Ground-truth annotation json.",
    )
    parser.add_argument(
        "--images",
        default="/l/users/mohamed.abouelhadid/SpiideoSynLoc/images/val",
        help="Validation image directory.",
    )
    parser.add_argument(
        "--pred-json",
        default="validation_results.json",
        help="Where to write the COCO-style detections json.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--width", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Detection confidence threshold.")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument(
        "--position-keypoint-index",
        type=int,
        default=1,
        help="Keypoint index containing the image-space player location. Default 1 = ground_pelvic.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Optional fixed score threshold. Leave unset on val to reproduce the devkit behavior.",
    )
    parser.add_argument(
        "--metrics-json",
        default="validation_metrics.json",
        help="Where to write the evaluation metrics json.",
    )
    return parser.parse_args()


def flatten_keypoints(keypoints_xy):
    flat = []
    for x, y in keypoints_xy:
        flat.extend([float(x), float(y), 1.0])
    return flat


def build_predictions(model_path: str, images_dir: str, gt_path: str, pred_json_path: str, args) -> None:
    from ultralytics import YOLO

    with open(gt_path) as f:
        gt = json.load(f)

    gt_images = gt["images"]
    image_id_by_name = {Path(img["file_name"]).name: img["id"] for img in gt["images"]}

    model = YOLO(model_path)
    results = model.predict(
        source=images_dir,
        imgsz=(args.height,args.width), #args.imgsz,
        conf=args.conf,
        device=args.device,
        max_det=args.max_det,
        save=False,
        verbose=False,
        stream=True,
        end2end=True
    )

    detections = []
    det_id = 1
    for result in tqdm(results, total=len(gt_images), desc="Running val inference", unit="image"):
        image_name = Path(result.path).name
        image_id = image_id_by_name.get(image_name)
        if image_id is None or result.boxes is None or result.keypoints is None:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        keypoints_xy = result.keypoints.xy.cpu().tolist()

        for box_xyxy, score, kps_xy in zip(boxes_xyxy, scores, keypoints_xy):
            x1, y1, x2, y2 = box_xyxy
            detections.append(
                {
                    "id": det_id,
                    "image_id": int(image_id),
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "keypoints": flatten_keypoints(kps_xy),
                    "score": float(score),
                }
            )
            det_id += 1

    with open(pred_json_path, "w") as f:
        json.dump(detections, f)

    print(f"Wrote {len(detections)} detections to {pred_json_path}")


def metrics_payload(coco_eval) -> dict:
    stats = [float(x) for x in coco_eval.stats]
    payload = {
        "stats": stats,
        "selected_metrics": {
            "mAP_LocSim": stats[0],
            "score_threshold": stats[15],
            "frame_accuracy": stats[16],
        },
    }

    if len(stats) >= 17:
        payload["named_stats"] = {
            "AP": stats[0],
            "AP_50": stats[1],
            "AP_75": stats[2],
            "AP_small": stats[3],
            "AP_medium": stats[4],
            "AP_large": stats[5],
            "AR_1": stats[6],
            "AR_10": stats[7],
            "AR_100": stats[8],
            "AR_small": stats[9],
            "AR_medium": stats[10],
            "AR_large": stats[11],
            "precision": stats[12],
            "recall": stats[13],
            "f1": stats[14],
            "score_threshold": stats[15],
            "frame_accuracy": stats[16],
        }

    return payload


def evaluate(
    gt_path: str,
    pred_json_path: str,
    position_keypoint_index: int,
    score_threshold,
    metrics_json_path: str,
):
    #repo_root = Path(__file__).resolve().parent
    #xtcocoapi_dir = repo_root / "xtcocoapi"
    #if str(xtcocoapi_dir) not in sys.path:
    #    sys.path.insert(0, str(xtcocoapi_dir))


    coco = COCO(gt_path)
    coco_det = coco.loadRes(pred_json_path)
    coco_eval = LocSimCOCOeval(coco, coco_det, "bbox", [0.089, 0.089], True)
    coco_eval.params.useSegm = None
    coco_eval.params.position_from_keypoint_index = position_keypoint_index
    if score_threshold is not None:
        coco_eval.params.score_threshold = score_threshold

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    payload = metrics_payload(coco_eval)
    with open(metrics_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSelected metrics")
    print(f"mAP-LocSim: {coco_eval.stats[0]:.4f}")
    print(f"score_threshold: {coco_eval.stats[15]:.6f}")
    print(f"frame_accuracy: {coco_eval.stats[16]:.4f}")
    print(f"Saved metrics to {metrics_json_path}")


def main() -> None:
    args = parse_args()
    build_predictions(args.model, args.images, args.gt, args.pred_json, args)
    evaluate(
        args.gt,
        args.pred_json,
        args.position_keypoint_index,
        args.score_threshold,
        args.metrics_json,
    )


if __name__ == "__main__":
    main()
