from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy import ndimage

if TYPE_CHECKING:
    import fitz


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTENSIONS = {".pdf"}
LABELS = [0, 90, 180, 270]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Prediction:
    angle: int
    confidence: float


@dataclass
class AggregatedPrediction:
    angle: int
    confidence: float
    votes: dict[int, float]


class OfflineOrientationClassifier:
    def __init__(self, model_dir: Path, use_gpu: bool = False) -> None:
        try:
            from paddle import inference
        except ImportError as exc:
            raise SystemExit(
                "paddlepaddle is not installed in the active Python environment. "
                "Install PaddlePaddle in your offline/conda environment before running this script."
            ) from exc

        model_file = model_dir / "inference.json"
        params_file = model_dir / "inference.pdiparams"

        if not model_file.exists() or not params_file.exists():
            raise SystemExit(
                f"Model files were not found in '{model_dir}'. "
                "Expected inference.json and inference.pdiparams."
            )

        self.predictor = self._create_predictor(
            inference=inference,
            model_file=model_file,
            params_file=params_file,
            use_gpu=use_gpu,
        )
        self.input_name = self.predictor.get_input_names()[0]
        self.output_name = self.predictor.get_output_names()[0]

    @staticmethod
    def _build_config(inference, model_file: Path, params_file: Path, use_gpu: bool, safe_mode: bool):
        config = inference.Config(str(model_file), str(params_file))
        if use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        config.switch_ir_optim(not safe_mode)
        if not safe_mode:
            config.enable_memory_optim()
        return config

    def _create_predictor(self, inference, model_file: Path, params_file: Path, use_gpu: bool):
        try:
            fast_config = self._build_config(
                inference=inference,
                model_file=model_file,
                params_file=params_file,
                use_gpu=use_gpu,
                safe_mode=False,
            )
            return inference.create_predictor(fast_config)
        except Exception as exc:
            warnings.warn(
                "Falling back to safe Paddle predictor settings because optimized predictor "
                f"creation failed: {exc}"
            )

        safe_config = self._build_config(
            inference=inference,
            model_file=model_file,
            params_file=params_file,
            use_gpu=use_gpu,
            safe_mode=True,
        )
        return inference.create_predictor(safe_config)

    def predict(self, image: Image.Image) -> Prediction:
        tensor = self._preprocess(image)
        input_handle = self.predictor.get_input_handle(self.input_name)
        input_handle.reshape(tensor.shape)
        input_handle.copy_from_cpu(tensor)
        self.predictor.run()

        output_handle = self.predictor.get_output_handle(self.output_name)
        logits = output_handle.copy_to_cpu()
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        probabilities = self._softmax(logits)
        class_id = int(np.argmax(probabilities))

        return Prediction(
            angle=LABELS[class_id],
            confidence=float(probabilities[class_id]),
        )

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size

        if width <= 0 or height <= 0:
            raise ValueError("Encountered an empty image while preprocessing.")

        short_side = min(width, height)
        scale = 256 / short_side
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = rgb_image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

        left = max(0, (resized_width - 224) // 2)
        top = max(0, (resized_height - 224) // 2)
        cropped = resized.crop((left, top, left + 224, top + 224))

        image_array = np.asarray(cropped, dtype=np.float32) / 255.0
        image_array = (image_array - MEAN) / STD
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)


def render_page(page: "fitz.Page", dpi: int) -> Image.Image:
    import fitz

    zoom = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def rotate_image_to_upright(image: Image.Image, angle: int) -> Image.Image:
    # PIL uses counter-clockwise angles. Paddle's labels represent the clockwise
    # offset from the upright orientation, so using the same numeric angle here
    # counter-rotates the image back to normal reading direction.
    return image.rotate(angle, expand=True)


def _white_fill(image: Image.Image) -> int | tuple[int, int, int, int]:
    bands = len(image.getbands())
    if bands == 1:
        return 255
    return tuple([255] * bands)


def _text_mask(image: Image.Image) -> np.ndarray:
    grayscale = image.convert("L")
    image_array = np.asarray(grayscale, dtype=np.float32)
    threshold = np.percentile(image_array, 35)
    return image_array < threshold


def _text_density(image: Image.Image) -> float:
    mask = _text_mask(image)
    return float(mask.mean())


def _generate_orientation_crops(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    if width < 224 or height < 224:
        return [image]

    crops: list[Image.Image] = [image]

    crop_specs = [
        (0.85, 0.85, 0.50, 0.50),
        (0.70, 0.70, 0.25, 0.25),
        (0.70, 0.70, 0.75, 0.25),
        (0.70, 0.70, 0.25, 0.75),
        (0.70, 0.70, 0.75, 0.75),
        (0.90, 0.45, 0.50, 0.25),
        (0.90, 0.45, 0.50, 0.50),
        (0.90, 0.45, 0.50, 0.75),
        (0.45, 0.90, 0.25, 0.50),
        (0.45, 0.90, 0.50, 0.50),
        (0.45, 0.90, 0.75, 0.50),
    ]

    for rel_w, rel_h, center_x, center_y in crop_specs:
        crop_width = max(224, int(round(width * rel_w)))
        crop_height = max(224, int(round(height * rel_h)))

        left = int(round(width * center_x - crop_width / 2))
        top = int(round(height * center_y - crop_height / 2))
        left = min(max(0, left), width - crop_width)
        top = min(max(0, top), height - crop_height)

        right = left + crop_width
        bottom = top + crop_height
        crops.append(image.crop((left, top, right, bottom)))

    return crops


def predict_orientation_with_voting(
    classifier: OfflineOrientationClassifier,
    image: Image.Image,
    min_text_density: float = 0.01,
) -> AggregatedPrediction:
    vote_totals = {angle: 0.0 for angle in LABELS}
    accepted_tiles = 0

    for crop in _generate_orientation_crops(image):
        density = _text_density(crop)
        if density < min_text_density:
            continue

        prediction = classifier.predict(crop)
        weight = max(density, 1e-6) * max(prediction.confidence, 1e-6)
        vote_totals[prediction.angle] += weight
        accepted_tiles += 1

    if accepted_tiles == 0:
        prediction = classifier.predict(image)
        vote_totals[prediction.angle] = prediction.confidence
        return AggregatedPrediction(
            angle=prediction.angle,
            confidence=prediction.confidence,
            votes=vote_totals,
        )

    best_angle = max(vote_totals, key=vote_totals.get)
    total_vote = sum(vote_totals.values())
    normalized_confidence = vote_totals[best_angle] / total_vote if total_vote else 0.0
    return AggregatedPrediction(
        angle=best_angle,
        confidence=float(normalized_confidence),
        votes=vote_totals,
    )


def estimate_skew_angle(
    image: Image.Image,
    max_angle: float = 15.0,
    coarse_step: float = 1.0,
    fine_step: float = 0.1,
) -> float:
    grayscale = image.convert("L")
    width, height = grayscale.size

    longest_side = max(width, height)
    if longest_side > 1600:
        scale = 1600 / longest_side
        resized = grayscale.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.BILINEAR,
        )
    else:
        resized = grayscale

    image_array = np.asarray(resized, dtype=np.float32)

    # Convert the page into a binary text mask. Documents typically have dark
    # foreground over a bright background, so lower intensities are treated as text.
    threshold = np.percentile(image_array, 35)
    binary = image_array < threshold
    if not np.any(binary):
        return 0.0

    binary_array = binary.astype(np.float32)

    def score_angle(angle: float) -> float:
        rotated = ndimage.rotate(
            binary_array,
            angle=angle,
            reshape=False,
            order=0,
            mode="constant",
            cval=0.0,
        )
        histogram = rotated.sum(axis=1)
        return float(np.var(histogram))

    coarse_angles = np.arange(-max_angle, max_angle + coarse_step, coarse_step)
    coarse_scores = [(score_angle(angle), angle) for angle in coarse_angles]
    _, best_coarse_angle = max(coarse_scores, key=lambda item: item[0])

    fine_start = max(-max_angle, best_coarse_angle - coarse_step)
    fine_end = min(max_angle, best_coarse_angle + coarse_step)
    fine_angles = np.arange(fine_start, fine_end + fine_step, fine_step)
    fine_scores = [(score_angle(angle), angle) for angle in fine_angles]
    _, best_fine_angle = max(fine_scores, key=lambda item: item[0])
    return round(float(best_fine_angle), 3)


def deskew_image(
    image: Image.Image,
    max_angle: float = 15.0,
    coarse_step: float = 1.0,
    fine_step: float = 0.1,
) -> tuple[Image.Image, float]:
    skew_angle = estimate_skew_angle(
        image,
        max_angle=max_angle,
        coarse_step=coarse_step,
        fine_step=fine_step,
    )
    if abs(skew_angle) < 0.05:
        return image, 0.0

    corrected = image.rotate(
        skew_angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=_white_fill(image),
    )
    return corrected, skew_angle


def save_pdf(images: list[Image.Image], output_path: Path) -> None:
    if not images:
        raise ValueError("No pages were generated for PDF export.")

    rgb_images = [image.convert("RGB") for image in images]
    first_image, remaining_images = rgb_images[0], rgb_images[1:]
    first_image.save(output_path, save_all=True, append_images=remaining_images, resolution=150.0)


def process_pdf(
    input_path: Path,
    output_dir: Path,
    classifier: OfflineOrientationClassifier,
    dpi: int,
    deskew_max_angle: float,
) -> dict:
    import fitz

    document = fitz.open(input_path)
    rotated_pages: list[Image.Image] = []
    page_results: list[dict] = []

    try:
        for page_index, page in enumerate(document, start=1):
            page_image = render_page(page, dpi)
            prediction = predict_orientation_with_voting(classifier, page_image)
            oriented_page = rotate_image_to_upright(page_image, prediction.angle)
            corrected_page, skew_angle = deskew_image(
                oriented_page,
                max_angle=deskew_max_angle,
            )
            rotated_pages.append(corrected_page)
            page_results.append(
                {
                    "page": page_index,
                    "predicted_angle": prediction.angle,
                    "confidence": round(prediction.confidence, 6),
                    "votes": {str(angle): round(score, 6) for angle, score in prediction.votes.items()},
                    "deskew_angle": skew_angle,
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
    classifier: OfflineOrientationClassifier,
    deskew_max_angle: float,
) -> dict:
    with Image.open(input_path) as image:
        prediction = predict_orientation_with_voting(classifier, image)
        oriented_image = rotate_image_to_upright(image, prediction.angle)
        rotated, skew_angle = deskew_image(oriented_image, max_angle=deskew_max_angle)
        output_path = output_dir / f"{input_path.stem}_rotated{input_path.suffix.lower()}"
        rotated.save(output_path)

    report = {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "type": "image",
        "predicted_angle": prediction.angle,
        "confidence": round(prediction.confidence, 6),
        "votes": {str(angle): round(score, 6) for angle, score in prediction.votes.items()},
        "deskew_angle": skew_angle,
    }
    save_report(output_dir / f"{input_path.stem}_rotation.json", report)
    return report


def save_report(report_path: Path, payload: dict) -> None:
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_input_files(input_dir: Path) -> list[Path]:
    supported_extensions = IMAGE_EXTENSIONS | PDF_EXTENSIONS
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in supported_extensions
    )


def build_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Rotate files from the Input folder using the local offline orientation model."
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
    parser.add_argument(
        "--deskew-max-angle",
        type=float,
        default=15.0,
        help="Maximum fine skew angle to search after the Paddle orientation correction.",
    )
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

    classifier = OfflineOrientationClassifier(model_dir=model_dir, use_gpu=args.use_gpu)
    summary: list[dict] = []

    for input_path in files:
        suffix = input_path.suffix.lower()
        if suffix in PDF_EXTENSIONS:
            result = process_pdf(
                input_path,
                output_dir,
                classifier,
                dpi=args.dpi,
                deskew_max_angle=args.deskew_max_angle,
            )
        elif suffix in IMAGE_EXTENSIONS:
            result = process_image(
                input_path,
                output_dir,
                classifier,
                deskew_max_angle=args.deskew_max_angle,
            )
        else:
            continue

        summary.append(result)
        print(f"Processed {input_path.name} -> {result['output_file']}")

    save_report(output_dir / "summary.json", {"processed_files": summary})
    print(f"Finished. Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
