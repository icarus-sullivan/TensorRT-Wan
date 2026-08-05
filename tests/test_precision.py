from tensorrt_wan.config.schema import PrecisionConfig
from tensorrt_wan.runtime.gpu import GPUArchitecture, GPUInfo
from tensorrt_wan.runtime.precision import select_precision


def _gpu(architecture: GPUArchitecture) -> GPUInfo:
    return GPUInfo(
        index=0,
        name="test-gpu",
        architecture=architecture,
        compute_capability=(9, 0),
        total_memory_bytes=1 << 34,
        cuda_version="12.4",
        driver_version="550.00",
    )


def test_blackwell_defaults_to_fp8():
    decision = select_precision(_gpu(GPUArchitecture.BLACKWELL), PrecisionConfig())
    assert decision.precision == "fp8"


def test_blackwell_respects_allow_fp8_false():
    decision = select_precision(_gpu(GPUArchitecture.BLACKWELL), PrecisionConfig(allow_fp8=False))
    assert decision.precision == "fp16"


def test_ampere_defaults_to_fp16():
    decision = select_precision(_gpu(GPUArchitecture.AMPERE), PrecisionConfig())
    assert decision.precision == "fp16"


def test_explicit_mode_overrides_auto_selection():
    decision = select_precision(_gpu(GPUArchitecture.BLACKWELL), PrecisionConfig(mode="fp16"))
    assert decision.precision == "fp16"
    assert "explicit" in decision.reason
