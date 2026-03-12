from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import fitz


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTENSIONS = {".pdf"}
LABELS = [0, 90, 180, 270]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_inference_module():
    try:
        from paddle import inference
    except ImportError as exc:
        raise SystemExit(
            "paddlepaddle is not installed in the active Python environment. "
            "Install PaddlePaddle before running this script."
        ) from exc
    return inference


def build_predictor_config(inference, model_file: Path, params_file: Path, use_gpu: bool, safe_mode: bool):
    config = inference.Config(str(model_file), str(params_file))
    if use_gpu:
        config.enable_use_gpu(1000, 0)
    else:
        config.disable_gpu()

    config.switch_ir_optim(not safe_mode)
    if not safe_mode:
        config.enable_memory_optim()
    return config


def create_predictor(model_dir: Path, use_gpu: bool = False):
    inference = load_inference_module()
    model_file = model_dir / "inference.json"
    params_file = model_dir / "inference.pdiparams"

    if not model_file.exists() or not params_file.exists():
        raise SystemExit(
            f"Model files were not found in '{model_dir}'. "
            "Expected inference.json and inference.pdiparams."
        )

    try:
        fast_config = build_predictor_config(
            inference=inference,
            model_file=model_file,
            params_file=params_file,
            use_gpu=use_gpu,
            safe_mode=False,
        )
        predictor = inference.create_predictor(fast_config)
    except Exception as exc:
        warnings.warn(
            "Falling back to safe Paddle predictor settings because optimized predictor "
            f"creation failed: {exc}"
        )
        safe_config = build_predictor_config(
            inference=inference,
            model_file=model_file,
            params_file=params_file,
            use_gpu=use_gpu,
            safe_mode=True,
        )
        predictor = inference.create_predictor(safe_config)

    return predictor, predictor.get_input_names()[0], predictor.get_output_names()[0]


def preprocess_image_for_model(image: Image.Image) -> np.ndarray:
    # Keep preprocessing simple: no extra crop generation or deskew logic.
    resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_array = (image_array - MEAN) / STD
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def predict_orientation(image: Image.Image, predictor, input_name: str, output_name: str) -> tuple[int, float]:
    tensor = preprocess_image_for_model(image)
    input_handle = predictor.get_input_handle(input_name)
    input_handle.reshape(tensor.shape)
    input_handle.copy_from_cpu(tensor)
    predictor.run()

    output_handle = predictor.get_output_handle(output_name)
    logits = np.asarray(output_handle.copy_to_cpu(), dtype=np.float32).reshape(-1)
    probabilities = softmax(logits)
    class_id = int(np.argmax(probabilities))
    return LABELS[class_id], float(probabilities[class_id])


def render_page(page: "fitz.Page", dpi: int) -> Image.Image:
    import fitz

    zoom = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def rotate_image_to_upright(image: Image.Image, angle: int) -> Image.Image:
    # PIL rotates counter-clockwise; using the predicted label corrects the page.
    return image.rotate(angle, expand=True)


def save_pdf(images: list[Image.Image], output_path: Path) -> None:
    if not images:
        raise ValueError("No pages were generated for PDF export.")

    rgb_images = [image.convert("RGB") for image in images]
    first_image, remaining_images = rgb_images[0], rgb_images[1:]
    first_image.save(output_path, save_all=True, append_images=remaining_images, resolution=150.0)


def save_report(report_path: Path, payload: dict) -> None:
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_input_files(input_dir: Path) -> list[Path]:
    supported_extensions = IMAGE_EXTENSIONS | PDF_EXTENSIONS
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in supported_extensions
    )


def process_pdf(
    input_path: Path,
    output_dir: Path,
    predictor,
    input_name: str,
    output_name: str,
    dpi: int,
) -> dict:
    import fitz

    document = fitz.open(input_path)
    rotated_pages: list[Image.Image] = []
    page_results: list[dict] = []

    try:
        for page_index, page in enumerate(document, start=1):
            page_image = render_page(page, dpi)
            angle, confidence = predict_orientation(page_image, predictor, input_name, output_name)
            rotated_page = rotate_image_to_upright(page_image, angle)
            rotated_pages.append(rotated_page)
            page_results.append(
                {
                    "page": page_index,
                    "predicted_angle": angle,
                    "confidence": round(confidence, 6),
                }
            )
    finally:
        document.close()

    output_path = output_dir / f"{input_path.stem}_rotated.pdf"
    save_pdf(rotated_pages, output_path)

    report = {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "type": "pdf",
        "pages": page_results,
    }
    save_report(output_dir / f"{input_path.stem}_rotation.json", report)
    return report


def process_image(
    input_path: Path,
    output_dir: Path,
    predictor,
    input_name: str,
    output_name: str,
) -> dict:
    with Image.open(input_path) as image:
        angle, confidence = predict_orientation(image, predictor, input_name, output_name)
        rotated = rotate_image_to_upright(image, angle)
        output_path = output_dir / f"{input_path.stem}_rotated{input_path.suffix.lower()}"
        rotated.save(output_path)

    report = {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "type": "image",
        "predicted_angle": angle,
        "confidence": round(confidence, 6),
    }
    save_report(output_dir / f"{input_path.stem}_rotation.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Rotate files from the Input folder using one orientation prediction per page/image."
    )
    parser.add_argument("--input-dir", type=Path, default=base_dir / "Input", help="Folder containing input files.")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "Output", help="Folder for rotated files.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=base_dir / "PP-LCNet_x1_0_doc_ori",
        help="Local model directory containing inference.json and inference.pdiparams.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI used for PDF page classification.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable Paddle GPU inference if available.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = args.model_dir.resolve()

    if not input_dir.exists():
        parser.error(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(input_dir)
    if not files:
        print(f"No supported files found in {input_dir}")
        return 0

    predictor, input_name, output_name = create_predictor(model_dir=model_dir, use_gpu=args.use_gpu)
    summary: list[dict] = []

    for input_path in files:
        suffix = input_path.suffix.lower()
        if suffix in PDF_EXTENSIONS:
            result = process_pdf(input_path, output_dir, predictor, input_name, output_name, dpi=args.dpi)
        elif suffix in IMAGE_EXTENSIONS:
            result = process_image(input_path, output_dir, predictor, input_name, output_name)
        else:
            continue

        summary.append(result)
        print(f"Processed {input_path.name} -> {result['output_file']}")

    save_report(output_dir / "summary.json", {"processed_files": summary})
    print(f"Finished. Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
