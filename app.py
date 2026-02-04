from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from preprocess import (
    PDF_MEDIA_TYPE,
    SUPPORTED_IMAGE_TYPES,
    preprocess_pdf,
    processed_image_bytes,
)

app = FastAPI(title="Document Preprocessing API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/preprocess")
async def preprocess_document(file: UploadFile = File(...)) -> StreamingResponse:
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    if file.content_type == PDF_MEDIA_TYPE:
        processed_pdf = preprocess_pdf(content)
        return StreamingResponse(
            io.BytesIO(processed_pdf),
            media_type=PDF_MEDIA_TYPE,
            headers={"Content-Disposition": "attachment; filename=processed.pdf"},
        )

    if file.content_type in SUPPORTED_IMAGE_TYPES:
        try:
            image = Image.open(io.BytesIO(content))
        except Exception as exc:  # pragma: no cover - defensive for invalid inputs
            raise HTTPException(
                status_code=400, detail="Invalid image file."
            ) from exc

        processed_bytes, ext = processed_image_bytes(image)
        return StreamingResponse(
            io.BytesIO(processed_bytes),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=processed.{ext}"},
        )

    raise HTTPException(
        status_code=415,
        detail="Unsupported file type. Upload an image or a PDF.",
    )
