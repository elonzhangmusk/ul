"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd

try:  # optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from ..dataset_tools.yolo_to_coco import convert_yolo_dataset_to_coco_json
except ImportError:  # pragma: no cover
    from dataset_tools.yolo_to_coco import convert_yolo_dataset_to_coco_json

try:  # optional dependency
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:  # pragma: no cover
    COCO = None
    COCOeval = None

try:  # optional dependency
    from thop import profile
except Exception:  # pragma: no cover
    profile = None


def _image_files(image_dir: Path) -> list[Path]:
    return sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])


def _resolve_label_path(label_dir: Path, image_path: Path, image_root: Path) -> Path:
    return (label_dir / image_path.relative_to(image_root)).with_suffix(".txt")


def _get_model_stats(yolo_model, img_size: int = 640) -> tuple[float, float]:
    if torch is None:
        return 0.0, 0.0
    model = yolo_model.model
    params = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = 0.0
    if profile is not None:
        try:
            device = next(model.parameters()).device
            dummy = torch.randn(1, 3, img_size, img_size, device=device)
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            gflops = float(flops) * 2.0 / 1e9
        except Exception:
            gflops = 0.0
    return float(params), float(gflops)


def _predict_to_coco_results(
    yolo_model: YOLO,
    image_files: list[Path],
    image_root: Path,
    img_size: int,
    conf: float,
    iou: float,
    device: str | int | None,
) -> tuple[list[dict], float]:
    predictions: list[dict] = []
    total_latency = 0.0
    for image_id, image_path in enumerate(image_files, start=1):
        start = time.perf_counter()
        results = yolo_model.predict(source=str(image_path), imgsz=img_size, conf=conf, iou=iou, device=device, save=False)
        total_latency += (time.perf_counter() - start) * 1000.0
        result = results[0]
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(box.cls[0].item()) + 1,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(x_max - x_min, 2), round(y_max - y_min, 2)],
                    "score": round(float(box.conf[0].item()), 6),
                }
            )
    avg_latency = total_latency / max(len(image_files), 1)
    return predictions, avg_latency


def evaluate_model(
    weights: str,
    image_dir: str | Path,
    label_dir: str | Path,
    class_names: Iterable[str],
    img_size: int = 640,
    conf: float = 0.001,
    iou: float = 0.65,
    device: str | int | None = None,
) -> dict[str, float | str]:
    """使用 COCOeval 评估单个权重。"""
    if COCO is None or COCOeval is None:
        raise ImportError("pycocotools is required for evaluation.")
    from ultralytics import YOLO

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    class_names = list(class_names)
    image_files = _image_files(image_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    yolo_model = YOLO(weights)
    if device is not None and torch is not None:
        try:
            device_str = f"cuda:{device}" if str(device).isdigit() else str(device)
            yolo_model.model.to(device_str)
        except Exception:
            pass

    params, gflops = _get_model_stats(yolo_model, img_size=img_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_json = Path(tmpdir) / "gt.json"
        pred_json = Path(tmpdir) / "pred.json"
        convert_yolo_dataset_to_coco_json(image_dir, label_dir, class_names, gt_json)

        predictions, avg_latency = _predict_to_coco_results(
            yolo_model=yolo_model,
            image_files=image_files,
            image_root=image_dir,
            img_size=img_size,
            conf=conf,
            iou=iou,
            device=device,
        )
        pred_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

        coco_gt = COCO(str(gt_json))
        if predictions:
            coco_dt = coco_gt.loadRes(str(pred_json))
        else:
            empty_pred = Path(tmpdir) / "empty_pred.json"
            empty_pred.write_text("[]", encoding="utf-8")
            coco_dt = coco_gt.loadRes(str(empty_pred))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats.tolist()

    result = {
        "model": Path(weights).stem,
        "AP_S": float(stats[3]),
        "AR_S": float(stats[9]),
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "Params(M)": params,
        "GFLOPs": gflops,
        "Latency(ms)": avg_latency,
        "FPS": 1000.0 / avg_latency if avg_latency > 0 else 0.0,
    }
    return result


def evaluate_models(
    weights: Iterable[str],
    image_dir: str | Path,
    label_dir: str | Path,
    class_names: Iterable[str],
    img_size: int = 640,
    conf: float = 0.001,
    iou: float = 0.65,
    device: str | int | None = None,
) -> pd.DataFrame:
    """批量评估多个权重并返回 DataFrame。"""
    rows = [
        evaluate_model(weight, image_dir, label_dir, class_names, img_size=img_size, conf=conf, iou=iou, device=device)
        for weight in weights
    ]
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate multiple YOLO weights using COCOeval.")
    parser.add_argument("--weights", nargs="+", required=True, help="权重文件列表")
    parser.add_argument("--img-dir", required=True, help="图像目录")
    parser.add_argument("--label-dir", required=True, help="标签目录")
    parser.add_argument("--class-names", nargs="+", default=["car"], help="类别名称列表")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.65, help="IoU 阈值")
    parser.add_argument("--device", default=None, help="设备编号")
    parser.add_argument("--out", default=None, help="CSV 输出路径")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = evaluate_models(
        args.weights,
        args.img_dir,
        args.label_dir,
        args.class_names,
        img_size=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
