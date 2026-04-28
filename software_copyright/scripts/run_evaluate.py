from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.evaluate_service import build_arg_parser, evaluate_models


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

