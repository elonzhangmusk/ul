"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2


def yolo_to_pixel(yolo_box: Iterable[float | str], img_w: int, img_h: int) -> tuple[int, int, int, int, int]:
    """将 YOLO 归一化框转换为像素坐标框。"""
    values = list(yolo_box)
    cls_id = int(float(values[0]))
    x_c, y_c, bw, bh = map(float, values[1:5])
    x_min = int(round((x_c - bw / 2.0) * img_w))
    y_min = int(round((y_c - bh / 2.0) * img_h))
    x_max = int(round((x_c + bw / 2.0) * img_w))
    y_max = int(round((y_c + bh / 2.0) * img_h))
    return cls_id, x_min, y_min, x_max, y_max


def pixel_to_yolo(
    cls_id: int,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    crop_w: int,
    crop_h: int,
) -> str:
    """将像素坐标框转换为单行 YOLO 文本。"""
    x_min = max(0, min(x_min, crop_w))
    y_min = max(0, min(y_min, crop_h))
    x_max = max(0, min(x_max, crop_w))
    y_max = max(0, min(y_max, crop_h))
    x_c = ((x_min + x_max) / 2.0) / crop_w
    y_c = ((y_min + y_max) / 2.0) / crop_h
    bw = (x_max - x_min) / crop_w
    bh = (y_max - y_min) / crop_h
    return f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"


def _image_files(image_dir: Path) -> list[Path]:
    return sorted(
        [p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )


def _label_path_for(image_path: Path, image_root: Path, label_root: Path) -> Path:
    relative = image_path.relative_to(image_root).with_suffix(".txt")
    return label_root / relative


def crop_dataset_split(
    image_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    crop_size: int = 640,
    overlap: float = 0.2,
    min_area_ratio: float = 0.4,
    keep_background: bool = False,
) -> dict[str, int]:
    """对图像目录执行切片，并同步裁剪 YOLO 标签。"""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    out_image_dir = output_dir / "images"
    out_label_dir = output_dir / "labels"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(round(crop_size * (1.0 - overlap))))
    saved_crops = 0
    background_crops = 0

    for image_path in _image_files(image_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]
        label_path = _label_path_for(image_path, image_dir, label_dir)
        boxes: list[tuple[int, int, int, int, int]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append(yolo_to_pixel(parts[:5], img_w, img_h))

        crop_width = min(crop_size, img_w)
        crop_height = min(crop_size, img_h)

        for y in range(0, max(img_h - crop_height + 1, 1), stride):
            for x in range(0, max(img_w - crop_width + 1, 1), stride):
                x_start = min(x, max(img_w - crop_width, 0))
                y_start = min(y, max(img_h - crop_height, 0))
                x_end = x_start + crop_width
                y_end = y_start + crop_height

                crop_img = image[y_start:y_end, x_start:x_end]
                crop_labels: list[str] = []

                for cls_id, bx_min, by_min, bx_max, by_max in boxes:
                    inter_x_min = max(x_start, bx_min)
                    inter_y_min = max(y_start, by_min)
                    inter_x_max = min(x_end, bx_max)
                    inter_y_max = min(y_end, by_max)
                    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
                        continue

                    orig_area = max((bx_max - bx_min) * (by_max - by_min), 1)
                    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                    if inter_area / orig_area < min_area_ratio:
                        continue

                    new_x_min = inter_x_min - x_start
                    new_y_min = inter_y_min - y_start
                    new_x_max = inter_x_max - x_start
                    new_y_max = inter_y_max - y_start
                    crop_labels.append(
                        pixel_to_yolo(cls_id, new_x_min, new_y_min, new_x_max, new_y_max, crop_width, crop_height)
                    )

                if not crop_labels and not keep_background:
                    continue

                relative_stem = image_path.relative_to(image_dir).with_suffix("")
                crop_name = f"{relative_stem.as_posix().replace('/', '_')}_{x_start}_{y_start}"
                out_image_path = out_image_dir / f"{crop_name}.jpg"
                out_label_path = out_label_dir / f"{crop_name}.txt"
                cv2.imwrite(str(out_image_path), crop_img)
                out_label_path.write_text("\n".join(crop_labels) + ("\n" if crop_labels else ""), encoding="utf-8")

                saved_crops += 1
                if not crop_labels:
                    background_crops += 1

    return {
        "saved_crops": saved_crops,
        "background_crops": background_crops,
        "crop_size": crop_size,
        "stride": stride,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop images and synchronize YOLO labels.")
    parser.add_argument("--image-dir", required=True, help="图像目录")
    parser.add_argument("--label-dir", required=True, help="标签目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--crop-size", type=int, default=640, help="切片尺寸")
    parser.add_argument("--overlap", type=float, default=0.2, help="切片重叠比例")
    parser.add_argument("--min-area-ratio", type=float, default=0.4, help="保留目标的最小交叠面积比例")
    parser.add_argument("--keep-background", action="store_true", help="是否保留背景切片")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = crop_dataset_split(
        args.image_dir,
        args.label_dir,
        args.output_dir,
        crop_size=args.crop_size,
        overlap=args.overlap,
        min_area_ratio=args.min_area_ratio,
        keep_background=args.keep_background,
    )
    print(summary)


if __name__ == "__main__":
    main()

