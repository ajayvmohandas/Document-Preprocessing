from __future__ import annotations

import io
from typing import Tuple

import cv2
import fitz
import numpy as np
from PIL import Image

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/tiff": "tiff",
    "image/bmp": "bmp",
    "image/webp": "webp",
}
PDF_MEDIA_TYPE = "application/pdf"
DPI = 300


def _pil_to_gray_np(image: Image.Image) -> np.ndarray:
    if image.mode != "L":
        image = image.convert("L")
    return np.array(image, dtype=np.uint8)


# Remove tiny speckles in a binary inverted mask.
def _remove_small_components(binary_inv: np.ndarray, min_area: int = 6) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    cleaned = np.zeros_like(binary_inv)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


# Estimate skew angle from text blobs and rotate.
def _deskew(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    cleaned = _remove_small_components(thresh, min_area=60)
    coords = np.column_stack(np.where(cleaned > 0))
    if coords.size == 0:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.3:
        return gray

    (height, width) = gray.shape[:2]
    center = (width // 2, height // 2)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        rotation,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


# Normalize uneven background illumination.
def _normalize_background(gray: np.ndarray) -> np.ndarray:
    background = cv2.medianBlur(gray, 25)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


# Enhance local contrast without over-sharpening.
def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
    return cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(denoised)


# Build a text mask for cleaning background noise.
def _build_clean_mask(gray: np.ndarray) -> np.ndarray:
    binary_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        6,
    )
    return _remove_small_components(binary_inv, min_area=6)


# Apply mask: keep text, whiten background.
def _apply_mask(gray: np.ndarray, mask_inv: np.ndarray) -> np.ndarray:
    output = gray.copy()
    output[mask_inv == 0] = 255
    return output


# Full preprocessing pipeline for a single image.
def preprocess_image(image: Image.Image) -> Image.Image:
    gray = _pil_to_gray_np(image)

    normalized = _normalize_background(gray)
    contrast = _enhance_contrast(normalized)
    deskewed = _deskew(contrast)
    cleaned_inv = _build_clean_mask(deskewed)
    output = _apply_mask(deskewed, cleaned_inv)

    return Image.fromarray(output, mode="L")


# Encode processed image as PNG bytes.
def processed_image_bytes(image: Image.Image) -> Tuple[bytes, str]:
    processed = preprocess_image(image)
    output = io.BytesIO()
    processed.save(output, format="PNG", compress_level=0)
    return output.getvalue(), "png"


# Split PDF into pages, preprocess, and merge.
def preprocess_pdf(pdf_bytes: bytes) -> bytes:
    source = fitz.open(stream=pdf_bytes, filetype="pdf")
    output_doc = fitz.open()

    try:
        for page in source:
            pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY, alpha=False)
            image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            processed_bytes, _ = processed_image_bytes(image)

            page_width = pix.width * 72 / DPI
            page_height = pix.height * 72 / DPI
            out_page = output_doc.new_page(width=page_width, height=page_height)
            out_page.insert_image(out_page.rect, stream=processed_bytes)

        return output_doc.write()
    finally:
        output_doc.close()
        source.close()
