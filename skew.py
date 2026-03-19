from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from PIL import Image

LABELS = [0, 90, 180, 270]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def create_inference_module():
    try:
        from paddle import inference
    except ImportError as exc:
        raise SystemExit(
            "paddlepaddle is not installed in the active Python environment. "
            "Install PaddlePaddle before running this script."
        ) from exc
    return inference


def validate_model_files(model_dir: Path) -> tuple[Path, Path]:
    model_file = model_dir / "inference.json"
    params_file = model_dir / "inference.pdiparams"
    if not model_file.exists() or not params_file.exists():
        raise SystemExit(
            f"Model files were not found in '{model_dir}'. "
            "Expected inference.json and inference.pdiparams."
        )
    return model_file, params_file


def build_config(inference, model_file: Path, params_file: Path, use_gpu: bool, safe_mode: bool):
    config = inference.Config(str(model_file), str(params_file))
    if use_gpu:
        config.enable_use_gpu(1000, 0)
    else:
        config.disable_gpu()

    config.switch_ir_optim(not safe_mode)
    if not safe_mode:
        config.enable_memory_optim()
    return config


def create_predictor(inference, model_file: Path, params_file: Path, use_gpu: bool):
    try:
        fast_config = build_config(
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

    safe_config = build_config(
        inference=inference,
        model_file=model_file,
        params_file=params_file,
        use_gpu=use_gpu,
        safe_mode=True,
    )
    return inference.create_predictor(safe_config)


def preprocess(image: Image.Image) -> np.ndarray:
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


def predict_angle(image: Image.Image, predictor, input_name: str, output_name: str) -> tuple[int, float]:
    tensor = preprocess(image)
    input_handle = predictor.get_input_handle(input_name)
    input_handle.reshape(tensor.shape)
    input_handle.copy_from_cpu(tensor)
    predictor.run()

    output_handle = predictor.get_output_handle(output_name)
    logits = np.asarray(output_handle.copy_to_cpu(), dtype=np.float32).reshape(-1)
    probabilities = softmax(logits)
    class_id = int(np.argmax(probabilities))
    angle = LABELS[class_id]
    confidence = float(probabilities[class_id])
    return angle, confidence


def rotate_image_to_upright(image: Image.Image, angle: int) -> Image.Image:
    return image.rotate(angle, expand=True)


PREDICTOR = None
INPUT_NAME = None
OUTPUT_NAME = None
CACHE_KEY = None


def get_predictor_handles(model_dir: Path, use_gpu: bool):
    global PREDICTOR, INPUT_NAME, OUTPUT_NAME, CACHE_KEY
    key = (model_dir.resolve(), use_gpu)
    if PREDICTOR is None or CACHE_KEY != key:
        inference = create_inference_module()
        model_file, params_file = validate_model_files(key[0])
        PREDICTOR = create_predictor(
            inference=inference,
            model_file=model_file,
            params_file=params_file,
            use_gpu=use_gpu,
        )
        INPUT_NAME = PREDICTOR.get_input_names()[0]
        OUTPUT_NAME = PREDICTOR.get_output_names()[0]
        CACHE_KEY = key
    return PREDICTOR, INPUT_NAME, OUTPUT_NAME


def image_skew_correction(
    image: Image.Image,
    model_dir: Path | str = Path(__file__).resolve().parent / "PP-LCNet_x1_0_doc_ori",
    use_gpu: bool = False,
) -> Image.Image:
    predictor, input_name, output_name = get_predictor_handles(Path(model_dir), use_gpu=use_gpu)
    angle, _confidence = predict_angle(image, predictor, input_name, output_name)
    return rotate_image_to_upright(image, angle)
