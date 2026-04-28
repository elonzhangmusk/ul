"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2


def _image_files(image_dir: Path) -> list[Path]:
    return sorted(
        [p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )


def convert_yolo_dataset_to_coco_json(
    image_dir: str | Path,
    label_dir: str | Path,
    class_names: Iterable[str],
    output_json: str | Path,
) -> dict:
    """将 YOLO 数据集转换为 COCO JSON。"""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_json = Path(output_json)
    class_names = list(class_names)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": idx + 1, "name": name} for idx, name in enumerate(class_names)],
    }

    annotation_id = 1
    for image_id, image_path in enumerate(_image_files(image_dir), start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        relative = image_path.relative_to(image_dir)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": relative.as_posix(),
                "width": width,
                "height": height,
            }
        )

        label_path = (label_dir / relative).with_suffix(".txt")
        if not label_path.exists():
            continue

        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0])) + 1
            x_c, y_c, bw, bh = map(float, parts[1:5])
            box_w = bw * width
            box_h = bh * height
            x_min = (x_c * width) - box_w / 2.0
            y_min = (y_c * height) - box_h / 2.0
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(box_w * box_h, 2),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    output_json.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return coco


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert YOLO labels to COCO JSON.")
    parser.add_argument("--image-dir", required=True, help="图像目录")
    parser.add_argument("--label-dir", required=True, help="标签目录")
    parser.add_argument("--class-names", nargs="+", required=True, help="类别名称列表")
    parser.add_argument("--output-json", required=True, help="输出 JSON 文件")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    convert_yolo_dataset_to_coco_json(args.image_dir, args.label_dir, args.class_names, args.output_json)
    print(f"COCO JSON saved to {args.output_json}")


if __name__ == "__main__":
    main()

