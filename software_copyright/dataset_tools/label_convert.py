"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def convert_bbox_to_yolo(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    img_width: int,
    img_height: int,
    class_id: int = 0,
) -> tuple[int, float, float, float, float]:
    """将像素坐标边界框转换为 YOLO 归一化格式。"""
    x_min = max(0.0, min(float(x_min), img_width))
    y_min = max(0.0, min(float(y_min), img_height))
    x_max = max(0.0, min(float(x_max), img_width))
    y_max = max(0.0, min(float(y_max), img_height))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid bbox coordinates.")

    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return class_id, x_center, y_center, width, height


def _resolve_class_id(
    raw_class: str,
    class_map: Mapping[str, int] | None,
) -> int:
    if class_map:
        if raw_class in class_map:
            return int(class_map[raw_class])
        if raw_class.isdigit() and str(int(raw_class)) in class_map:
            return int(class_map[str(int(raw_class))])
    if raw_class.isdigit():
        return int(raw_class)
    raise KeyError(f"Class token '{raw_class}' is not covered by class_map.")


def _parse_raw_label_line(
    line: str,
    class_map: Mapping[str, int] | None,
) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().replace(",", " ").split()
    if len(parts) < 5:
        return None

    if len(parts) >= 6 and _is_number(parts[1]):
        raw_class = parts[0]
        coords = list(map(float, parts[1:5]))
    elif _is_number(parts[0]):
        raw_class = parts[0]
        coords = list(map(float, parts[1:5]))
    else:
        raw_class = parts[0]
        coords = list(map(float, parts[1:5]))

    class_id = _resolve_class_id(raw_class, class_map)
    return class_id, *coords


def convert_label_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    img_width: int,
    img_height: int,
    class_map: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """批量将原始标签目录转换为 YOLO 标注目录。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_files = 0
    converted_boxes = 0
    skipped_boxes = 0

    for txt_path in sorted(input_dir.rglob("*.txt")):
        relative = txt_path.relative_to(input_dir)
        out_path = output_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)

        yolo_lines: list[str] = []
        for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = _parse_raw_label_line(line, class_map)
            if parsed is None:
                skipped_boxes += 1
                continue
            class_id, x_min, y_min, x_max, y_max = parsed
            try:
                yolo_box = convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height, class_id)
            except ValueError:
                skipped_boxes += 1
                continue
            yolo_lines.append(
                f"{yolo_box[0]} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}"
            )
            converted_boxes += 1

        out_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        converted_files += 1

    summary = {
        "converted_files": converted_files,
        "converted_boxes": converted_boxes,
        "skipped_boxes": skipped_boxes,
    }
    return summary


def _parse_class_map(values: Iterable[str] | None) -> dict[str, int] | None:
    if not values:
        return None
    class_map: dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid class mapping '{item}'. Use NAME=ID or ID=ID.")
        key, value = item.split("=", 1)
        class_map[key.strip()] = int(value.strip())
    return class_map


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert raw labels to YOLO format.")
    parser.add_argument("--input-dir", required=True, help="原始标签目录")
    parser.add_argument("--output-dir", required=True, help="输出 YOLO 标签目录")
    parser.add_argument("--img-width", type=int, required=True, help="图像宽度")
    parser.add_argument("--img-height", type=int, required=True, help="图像高度")
    parser.add_argument(
        "--class-map",
        nargs="*",
        help="类别映射，例如 car=0 bus=1。若原始标签已为数字类别，也可直接写 0=0。",
    )
    parser.add_argument(
        "--class-map-json",
        help="可选，类别映射 JSON 文件路径，内容应为 {\"car\": 0}",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    class_map = _parse_class_map(args.class_map)
    if args.class_map_json:
        class_map = dict(class_map or {})
        class_map.update(json.loads(Path(args.class_map_json).read_text(encoding="utf-8")))

    summary = convert_label_dir(args.input_dir, args.output_dir, args.img_width, args.img_height, class_map)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

