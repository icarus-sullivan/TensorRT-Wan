"""`_decomposed_rms_norm` (examples/loaders/wan_comfyui_loader.py) must match
`torch.nn.functional.rms_norm`'s real output -- it's a monkeypatch substitute for that exact
function during a MIGraphX-targeted export (opset 19 has no native RMSNormalization op), so any
numerical divergence here would silently change what the exported DiT graph computes. Pure CPU
math, no GPU/ROCm/MIGraphX required -- see docs/rocm_setup.md for what does need real hardware.
"""

import torch

from examples.loaders.wan_comfyui_loader import _decomposed_rms_norm


def test_matches_native_with_weight():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, dtype=torch.float32)
    weight = torch.randn(16, dtype=torch.float32)
    expected = torch.nn.functional.rms_norm(x, (16,), weight, eps=1e-6)
    actual = _decomposed_rms_norm(x, (16,), weight, eps=1e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_matches_native_without_weight():
    torch.manual_seed(1)
    x = torch.randn(3, 4, dtype=torch.float32)
    expected = torch.nn.functional.rms_norm(x, (4,), None, eps=1e-5)
    actual = _decomposed_rms_norm(x, (4,), None, eps=1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_matches_native_default_eps():
    torch.manual_seed(2)
    x = torch.randn(2, 4, dtype=torch.float32)
    expected = torch.nn.functional.rms_norm(x, (4,))
    actual = _decomposed_rms_norm(x, (4,))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_accepts_int_normalized_shape():
    # comfy.ops's RMSNorm (a torch.nn.RMSNorm subclass) may pass normalized_shape as either a
    # plain int or a tuple depending on how it was constructed -- both must work.
    torch.manual_seed(3)
    x = torch.randn(2, 4, dtype=torch.float32)
    assert torch.equal(_decomposed_rms_norm(x, 4), _decomposed_rms_norm(x, (4,)))
