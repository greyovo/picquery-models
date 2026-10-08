#!/usr/bin/env python3
"""Export MobileCLIP2 S0/S2/S3 text (dynamic INT8) and visual (FP16) for PicQuery.

Visual I/O and text output stay FP32. Text accepts INT64 token IDs, matching
PicQuery ort_engine.dart.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime import quantization
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel

MODEL_ROOT = Path(__file__).resolve().parent
ROOT = MODEL_ROOT.parents[1]
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(MODEL_ROOT / "third_party/open_clip/src"))

import open_clip  # noqa: E402
from mobileclip.modules.common.mobileone import reparameterize_model  # noqa: E402

MODELS = {
    "s0": ("MobileCLIP2-S0", "dfndr2b"),
    "s2": ("MobileCLIP2-S2", "dfndr2b"),
    "s3": ("MobileCLIP2-S3", "dfndr2b"),
}


def load_reference(size: str = "s0", checkpoint: Path | None = None):
    """Load the original FP32 encoder and its official preprocessing/tokenizer."""
    name, tag = MODELS[size]
    local = MODEL_ROOT / "checkpoints" / f"{name.lower().replace('-', '_')}.pt"
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    pretrained = str(checkpoint or local) if checkpoint or local.is_file() else tag
    # S0/S2 consume RGB in [0,1]; S3 uses the official CLIP normalization.
    preprocess_config = open_clip.get_pretrained_cfg(name, tag)
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, device="cpu",
        image_mean=preprocess_config["mean"], image_std=preprocess_config["std"],
        image_interpolation=preprocess_config["interpolation"],
        image_resize_mode=preprocess_config["resize_mode"],
    )
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(name)


def export_fp32(module, sample, path: Path, input_name: str, output_name: str):
    # The native MHA fast path has no legacy ONNX symbolic. Disable it only
    # during export and restore the caller's setting afterwards.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                module, sample, path, export_params=True, external_data=False,
                opset_version=18, dynamo=False, do_constant_folding=True,
                input_names=[input_name], output_names=[output_name],
                dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def export_visual(module, sample, path: Path):
    with tempfile.TemporaryDirectory(prefix=".visual_", dir=path.parent) as tmp:
        fp32 = Path(tmp) / "visual.onnx"
        export_fp32(module, sample, fp32, "image", "image_features")
        converted = convert_float_to_float16(onnx.load(fp32), keep_io_types=True)
        # The converter appends I/O Cast nodes; restore dependency order.
        OnnxModel(converted).topological_sort()
        onnx.checker.check_model(converted)
        onnx.save_model(converted, path, save_as_external_data=False)


def export_text(module, sample, path: Path):
    with tempfile.TemporaryDirectory(prefix=".text_", dir=path.parent) as tmp:
        fp32 = Path(tmp) / "text.onnx"
        preprocessed = Path(tmp) / "text_preprocessed.onnx"
        export_fp32(module, sample, fp32, "text", "text_features")
        quantization.quant_pre_process(fp32, preprocessed, skip_symbolic_shape=True)
        quantization.quantize_dynamic(
            preprocessed, path, per_channel=True,
            weight_type=quantization.QuantType.QInt8,
            # Quantize matrix weights and embedding tables; leave norm/bias FP32.
            op_types_to_quantize=["MatMul", "Gather"],
        )
        onnx.checker.check_model(onnx.load(path))


def validate_artifact(path: Path, sample: torch.Tensor, expected_dtype: int, embedding_dim: int):
    graph = onnx.load(path)
    assert graph.graph.input[0].type.tensor_type.elem_type == expected_dtype
    assert graph.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 2):
        value = sample.repeat(batch, *([1] * (sample.ndim - 1))).numpy()
        result = session.run(None, {session.get_inputs()[0].name: value})[0]
        if result.dtype != np.float32 or result.shape != (batch, embedding_dim) or not np.isfinite(result).all():
            raise RuntimeError(f"Invalid inference result from {path}: {result.shape}, {result.dtype}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS, nargs="?", default="s0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model, _, tokenizer = load_reference(args.model, args.checkpoint)
    model = reparameterize_model(model).eval()
    config = open_clip.get_model_config(MODELS[args.model][0])
    size = config["vision_cfg"]["image_size"]
    size = (size, size) if isinstance(size, int) else size
    image = torch.zeros(1, 3, *size, dtype=torch.float32)
    tokens = tokenizer(["a photograph of a cat"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = MODELS[args.model][0].lower().replace("-", "_")
    # Translation is reserved for: mt_zho-eng.fp32.quantized.onnx.
    # Stage and validate both files before replacing existing final artifacts.
    with tempfile.TemporaryDirectory(prefix=".export_", dir=args.output_dir) as tmp:
        staged = Path(tmp)
        text = staged / f"{prefix}_text.onnx"
        visual = staged / f"{prefix}_visual.onnx"
        print(f"Exporting {text.name} (INT8) ...", flush=True)
        export_text(model.text, tokens, text)
        print(f"Exporting {visual.name} (FP16, FP32 I/O) ...", flush=True)
        export_visual(model.visual, image, visual)
        validate_artifact(text, tokens, onnx.TensorProto.INT64, config["embed_dim"])
        validate_artifact(visual, image, onnx.TensorProto.FLOAT, config["embed_dim"])
        for artifact in (text, visual):
            target = args.output_dir / artifact.name
            artifact.replace(target)
            print(f"{target.resolve()} ({target.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
