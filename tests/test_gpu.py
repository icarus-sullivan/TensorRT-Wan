"""GPU detection tests that must pass with or without a real GPU/CUDA present — this is the one
module in the runtime layer explicitly designed to import and run cleanly on a CPU-only machine.
"""

from tensorrt_wan.runtime.gpu import (
    GPUArchitecture,
    _classify,
    _classify_amd,
    detect_gpus,
    is_cuda_available,
    is_rocm_available,
)


def test_detect_gpus_never_raises_without_cuda():
    # Must not raise even when torch/CUDA aren't available; an empty list is the valid result.
    gpus = detect_gpus()
    assert isinstance(gpus, list)
    if not is_cuda_available():
        assert gpus == []


def test_is_rocm_available_never_raises_without_rocm():
    # Same "must not raise, and a CUDA-less machine implies no ROCm either" contract as
    # test_detect_gpus_never_raises_without_cuda above.
    result = is_rocm_available()
    assert isinstance(result, bool)
    if not is_cuda_available():
        assert result is False


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


def test_classify_amd_rdna3():
    assert _classify_amd("gfx1103") == GPUArchitecture.AMD_RDNA3
    assert _classify_amd("gfx1100") == GPUArchitecture.AMD_RDNA3


def test_classify_amd_unknown_falls_back():
    assert _classify_amd("gfx900") == GPUArchitecture.AMD_UNKNOWN
    assert _classify_amd("") == GPUArchitecture.AMD_UNKNOWN
