"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def fix_class_ids(label_dir: str | Path, target_class_id: int = 0) -> dict[str, int]:
    """递归修复标签目录中的类别编号。"""
    label_dir = Path(label_dir)
    total_files = 0
    modified_files = 0
    modified_lines = 0

    for txt_path in sorted(label_dir.rglob("*.txt")):
        total_files += 1
        lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        new_lines: list[str] = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[0] != str(target_class_id):
                parts[0] = str(target_class_id)
                changed = True
                modified_lines += 1
            new_lines.append(" ".join(parts))
        if changed:
            txt_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
            modified_files += 1

    return {
        "total_files": total_files,
        "modified_files": modified_files,
        "modified_lines": modified_lines,
        "target_class_id": target_class_id,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fix class ids in YOLO label files.")
    parser.add_argument("--label-dir", required=True, help="标签目录")
    parser.add_argument("--target-class-id", type=int, default=0, help="目标类别编号")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(fix_class_ids(args.label_dir, args.target_class_id))


if __name__ == "__main__":
    main()

