from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.predict_service import build_arg_parser, predict_images


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

