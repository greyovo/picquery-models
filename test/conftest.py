"""Shared MobileCLIP integration fixtures; missing artifacts fail explicitly."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models/ml-mobileclip"
spec = importlib.util.spec_from_file_location("mobileclip_export", MODEL_ROOT / "export_mobileclip_models.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

TEXTS = [
    "a photograph of a cat", "a dog running in a park", "a red car on a road",
    "a mountain landscape with snow", "a diagram", "a bowl of fresh fruit",
    "a person standing by the ocean", "", "1234567890", "Hello, world!",
    "a " * 100, "a screenshot of a mobile application",
]


def pytest_addoption(parser):
    parser.addoption("--model", choices=list(exporter.MODELS), default="s0", help="MobileCLIP2 variant")
    parser.addoption("--checkpoint", type=Path, default=None, help="Original MobileCLIP2 FP32 checkpoint")
    parser.addoption("--outputs", type=Path, default=ROOT / "outputs", help="Directory containing exported MobileCLIP2 models")


@pytest.fixture(scope="session")
def reference(pytestconfig):
    size = pytestconfig.getoption("--model")
    checkpoint = pytestconfig.getoption("--checkpoint") or MODEL_ROOT / f"checkpoints/mobileclip2_{size}.pt"
    if not checkpoint.is_file():
        pytest.fail(f"Original checkpoint is required: {checkpoint}; supply --checkpoint PATH.")
    torch.set_num_threads(4)
    model, preprocess, tokenizer = exporter.load_reference(size, checkpoint)
    # Deliberately keep the original model BEFORE deployment reparameterization.
    return model, preprocess, tokenizer


@pytest.fixture(scope="session")
def inputs(reference):
    _, preprocess, tokenizer = reference
    images = []
    for name in ("example.png", "diagram-square.png", "fig_accuracy_latency.png"):
        with Image.open(MODEL_ROOT / "docs" / name) as image:
            images.append(preprocess(image.convert("RGB")))
    for value in (0, 127, 255):
        images.append(preprocess(Image.new("RGB", (256, 256), (value, value, value))))
    random = np.random.default_rng(42).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    images.append(preprocess(Image.fromarray(random)))
    return torch.stack(images), tokenizer(TEXTS)


@pytest.fixture(scope="session")
def sessions(pytestconfig):
    result = {}
    directory = pytestconfig.getoption("--outputs")
    size = pytestconfig.getoption("--model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    for encoder in ("text", "visual"):
        path = directory / f"mobileclip2_{size}_{encoder}.onnx"
        if not path.is_file():
            pytest.fail(f"Missing model {path}; run ./export.sh {size} first.")
        result[encoder] = (path, ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"]))
    return result


@pytest.fixture(scope="session")
def embeddings(reference, inputs, sessions):
    model, _, _ = reference
    image, text = inputs
    result = {}
    with torch.inference_mode():
        for encoder, tensor, function in (("text", text, model.encode_text), ("visual", image, model.encode_image)):
            expected = function(tensor).numpy()
            session = sessions[encoder][1]
            actual = session.run(None, {session.get_inputs()[0].name: tensor.numpy()})[0]
            result[encoder] = (expected, actual)
    return result


@pytest.fixture(scope="session")
def embedding_dim(pytestconfig):
    name = exporter.MODELS[pytestconfig.getoption("--model")][0]
    return exporter.open_clip.get_model_config(name)["embed_dim"]
