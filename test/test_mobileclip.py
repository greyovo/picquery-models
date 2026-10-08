"""Validate app I/O, parameter storage, inference and FP32 reference accuracy."""
import numpy as np
import onnx
import pytest


def normalize(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(min=1e-12)


@pytest.mark.parametrize("encoder,input_type,shape", [
    ("text", "tensor(int64)", ["batch", 77]),
    ("visual", "tensor(float)", ["batch", 3, 256, 256]),
])
def test_app_io_contract(sessions, embedding_dim, encoder, input_type, shape):
    _, session = sessions[encoder]
    assert len(session.get_inputs()) == len(session.get_outputs()) == 1
    assert session.get_inputs()[0].type == input_type
    assert session.get_inputs()[0].shape == shape
    assert session.get_outputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].shape == ["batch", embedding_dim]


@pytest.mark.parametrize("encoder,storage", [("text", onnx.TensorProto.INT8), ("visual", onnx.TensorProto.FLOAT16)])
def test_compressed_parameters(sessions, reference, encoder, storage):
    path, _ = sessions[encoder]
    model = onnx.load(path)
    onnx.checker.check_model(model)
    # Count parameter elements, rather than tensors: small FP32 biases/norms and
    # INT64 shape constants are expected to remain in a compressed model.
    floating = [x for x in model.graph.initializer if x.data_type in (
        onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.INT8,
    )]
    total = sum(int(np.prod(x.dims)) for x in floating)
    compressed = sum(int(np.prod(x.dims)) for x in floating if x.data_type == storage)
    assert compressed / total > 0.95
    # Scale the size bound to each architecture's original FP32 parameters.
    module = reference[0].text if encoder == "text" else reference[0].visual
    original_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    assert path.stat().st_size < original_bytes * (0.35 if encoder == "text" else 0.60)


@pytest.mark.parametrize("encoder", ["text", "visual"])
def test_single_and_batch_inference(sessions, inputs, embeddings, embedding_dim, encoder):
    _, session = sessions[encoder]
    values = inputs[1 if encoder == "text" else 0].numpy()
    actual = embeddings[encoder][1]
    assert actual.shape == (len(values), embedding_dim)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert np.all(np.linalg.norm(actual, axis=1) > 0)
    single = np.concatenate([
        session.run(None, {session.get_inputs()[0].name: value[None]})[0]
        for value in values
    ])
    # Dynamic activation scales can differ between INT8 single/batch execution.
    cosine = np.sum(normalize(single) * normalize(actual), axis=-1)
    assert cosine.min() >= (0.99 if encoder == "text" else 0.9999)


@pytest.mark.parametrize("encoder,min_cosine,max_relative_l2", [
    ("text", 0.99, 0.15), ("visual", 0.9998, 0.02),
])
def test_accuracy_against_original_fp32(embeddings, pytestconfig, encoder, min_cosine, max_relative_l2):
    # Keep S0/S2 budgets; the larger S3 visual encoder has a 3% L2 budget.
    if encoder == "visual" and pytestconfig.getoption("--model") == "s3":
        min_cosine, max_relative_l2 = 0.9995, 0.03
    expected, actual = embeddings[encoder]
    cosine = np.sum(normalize(expected) * normalize(actual), axis=-1)
    relative = np.linalg.norm(actual - expected, axis=-1) / np.linalg.norm(expected, axis=-1)
    print(f"\n{encoder}: min cosine={cosine.min():.6f}, max relative L2={relative.max():.6f}")
    assert cosine.min() >= min_cosine
    assert relative.max() <= max_relative_l2


def test_image_text_similarity_accuracy(embeddings):
    text_ref, text = embeddings["text"]
    visual_ref, visual = embeddings["visual"]
    expected = normalize(visual_ref) @ normalize(text_ref).T
    actual = normalize(visual) @ normalize(text).T
    error = np.max(np.abs(actual - expected))
    print(f"\nimage/text: max cosine-score error={error:.6f}")
    assert error <= 0.025
    # Assert top-1 only where the FP32 result has a margin above error tolerance.
    sorted_scores = np.sort(expected, axis=-1)
    confident = sorted_scores[:, -1] - sorted_scores[:, -2] > 0.05
    np.testing.assert_array_equal(actual.argmax(axis=-1)[confident], expected.argmax(axis=-1)[confident])
