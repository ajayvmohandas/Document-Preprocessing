from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from skew import image_skew_correction


def build_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Apply skew/orientation correction to a single image using Paddle model."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to input image file.")
    parser.add_argument("--output", type=Path, default=base_dir / "output_skew_corrected.png", help="Path to save corrected image.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=base_dir / "PP-LCNet_x1_0_doc_ori",
        help="Directory containing inference.json and inference.pdiparams.",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Enable Paddle GPU inference if available.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    model_dir = args.model_dir.resolve()

    if not input_path.exists():
        parser.error(f"Input image does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        corrected = image_skew_correction(image=image, model_dir=model_dir, use_gpu=args.use_gpu)
        corrected.save(output_path)

    print(f"Skew-corrected image saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
