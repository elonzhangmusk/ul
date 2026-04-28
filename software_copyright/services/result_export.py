"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def export_dataframe_to_csv(data: pd.DataFrame | list[Mapping[str, Any]], output_csv: str | Path) -> Path:
    """导出评估结果为 CSV 文件。"""
    output_csv = Path(output_csv)
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, Mapping):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(list(data))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_csv


def save_config_and_metrics(config: Mapping[str, Any], metrics: Mapping[str, Any], output_json: str | Path) -> Path:
    """将配置和指标保存为 JSON 文件。"""
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": dict(config), "metrics": dict(metrics)}
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export metrics to CSV/JSON.")
    parser.add_argument("--csv", help="CSV 输出路径")
    parser.add_argument("--json", help="JSON 输出路径")
    parser.add_argument("--payload", help="可选 JSON 字符串或 JSON 文件路径")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload: dict[str, Any] = {"config": {}, "metrics": {}}
    if args.payload:
        payload_path = Path(args.payload)
        if payload_path.exists():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(args.payload)

    if args.csv:
        export_dataframe_to_csv(payload.get("metrics", {}), args.csv)
    if args.json:
        save_config_and_metrics(payload.get("config", {}), payload.get("metrics", {}), args.json)


if __name__ == "__main__":
    main()
