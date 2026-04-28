from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.train_service import build_arg_parser, train_model


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

