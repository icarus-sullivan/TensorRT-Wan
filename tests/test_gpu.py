"""GPU detection tests that must pass with or without a real GPU/CUDA present — this is the one
module in the runtime layer explicitly designed to import and run cleanly on a CPU-only machine.
"""

from tensorrt_wan.runtime.gpu import GPUArchitecture, _classify, detect_gpus, is_cuda_available


def test_detect_gpus_never_raises_without_cuda():
    # Must not raise even when torch/CUDA aren't available; an empty list is the valid result.
    gpus = detect_gpus()
    assert isinstance(gpus, list)
    if not is_cuda_available():
        assert gpus == []


def test_classify_known_architectures():
    assert _classify(8, 0) == GPUArchitecture.AMPERE
    assert _classify(8, 9) == GPUArchitecture.ADA
    assert _classify(7, 5) == GPUArchitecture.TURING
    assert _classify(7, 0) == GPUArchitecture.VOLTA
    assert _classify(9, 0) == GPUArchitecture.HOPPER
    assert _classify(10, 0) == GPUArchitecture.BLACKWELL
    assert _classify(12, 0) == GPUArchitecture.BLACKWELL


def test_classify_unknown_falls_back():
    assert _classify(3, 0) == GPUArchitecture.UNKNOWN
