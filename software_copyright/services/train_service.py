"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

import argparse
from typing import Any


def train_model(
    model_cfg: str,
    data: str,
    pretrained_weights: str | None = None,
    imgsz: int = 640,
    epochs: int = 100,
    batch: int = 16,
    workers: int = 8,
    device: str | int | None = None,
    project: str | None = None,
    name: str | None = None,
    **kwargs: Any,
):
    """封装 YOLO 训练流程。"""
    from ultralytics import YOLO

    model = YOLO(model_cfg)
    if pretrained_weights:
        model.load(pretrained_weights)
    return model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        device=device,
        project=project,
        name=name,
        **kwargs,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO model for UAV small-object detection.")
    parser.add_argument("--model", required=True, help="模型配置文件")
    parser.add_argument("--data", required=True, help="数据集 YAML")
    parser.add_argument("--weights", help="预训练权重")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--device", default=None, help="设备编号")
    parser.add_argument("--project", default=None, help="输出目录")
    parser.add_argument("--name", default=None, help="实验名称")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_model(
        model_cfg=args.model,
        data=args.data,
        pretrained_weights=args.weights,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
