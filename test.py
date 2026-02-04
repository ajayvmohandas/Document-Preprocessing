from __future__ import annotations

from pathlib import Path

from PIL import Image

from preprocess import preprocess_pdf, processed_image_bytes

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def main() -> None:
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in INPUT_DIR.iterdir():
        if not file_path.is_file():
            continue

        if _is_pdf(file_path):
            pdf_bytes = file_path.read_bytes()
            processed = preprocess_pdf(pdf_bytes)
            output_path = OUTPUT_DIR / f"{file_path.stem}_processed.pdf"
            output_path.write_bytes(processed)
            print(f"Processed PDF: {file_path.name} -> {output_path.name}")
            continue

        if _is_image(file_path):
            with Image.open(file_path) as image:
                processed_bytes, _ = processed_image_bytes(image)

            output_path = OUTPUT_DIR / f"{file_path.stem}_processed.png"
            output_path.write_bytes(processed_bytes)
            print(f"Processed image: {file_path.name} -> {output_path.name}")
            continue

        print(f"Skipped unsupported file: {file_path.name}")


if __name__ == "__main__":
    main()
