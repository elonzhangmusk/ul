"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

import argparse
from typing import Any


def predict_images(
    weights: str,
    source: str,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | int | None = None,
    project: str | None = None,
    name: str | None = None,
    save: bool = True,
    **kwargs: Any,
):
    """封装 YOLO 推理流程。"""
    from ultralytics import YOLO

    model = YOLO(weights)
    return model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        save=save,
        **kwargs,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO inference on images or folders.")
    parser.add_argument("--weights", required=True, help="模型权重")
    parser.add_argument("--source", required=True, help="图片或文件夹路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--device", default=None, help="设备编号")
    parser.add_argument("--project", default=None, help="输出目录")
    parser.add_argument("--name", default=None, help="实验名称")
    parser.add_argument("--no-save", action="store_true", help="是否不保存可视化结果")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    predict_images(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()
