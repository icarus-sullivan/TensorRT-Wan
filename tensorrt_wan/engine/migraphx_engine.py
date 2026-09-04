"""MIGraphX-backed counterpart to `engine/base.py`'s `TensorRTEngineWrapper` — loads a static
(single-resolution) ONNX file under AMD's MIGraphX execution provider and runs it, for hardware
with no TensorRT (see `export/migraphx_build.py`'s module docstring for the full rationale).

Deliberately does *not* share a base class with `TensorRTEngineWrapper`: the two backends'
construction-time concerns (TensorRT's LoRA refit / multi-resolution optimization profiles vs.
MIGraphX's single fixed-shape static program, no refit support) are genuinely different, not just
an implementation detail. What they *do* share is the duck-typed `.load()`/`.unload()`/
`.infer(dict[str, Tensor]) -> dict[str, Tensor]` surface `DiTEngine` (`engine/dit_engine.py`)
depends on — pass an instance of this class as `DiTEngine(..., wrapper=MIGraphXEngineWrapper(...))`
to reuse `DiTEngine`'s conditioning-assembly logic unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from tensorrt_wan.export.migraphx_build import PRECISION_TO_MIGRAPHX_FLAG
from tensorrt_wan.runtime.fallback import run_with_fallback
from tensorrt_wan.utils.logging import get_logger

if TYPE_CHECKING:
    import onnxruntime as ort

logger = get_logger(__name__)


class MIGraphXEngineWrapper:
    """Loads a static-shape `.onnx` file (cached by `export/migraphx_build.py`, one per
    resolution profile — see `DiTExporter`'s `static=True`) and runs it under
    `MIGraphXExecutionProvider`.

    `torch_fallback`, if given, is called via `runtime.fallback.run_with_fallback` on any
    MIGraphX failure, matching `TensorRTEngineWrapper`'s existing automatic-fallback rule.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        device: torch.device | None = None,
        precision: str = "bf16",
        torch_fallback: "torch.nn.Module | None" = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.device = device or torch.device("cuda")
        if precision not in PRECISION_TO_MIGRAPHX_FLAG:
            raise ValueError(
                f"precision={precision!r} has no MIGraphX provider-option equivalent; "
                f"want one of {sorted(PRECISION_TO_MIGRAPHX_FLAG)}"
            )
        self.precision = precision
        self.torch_fallback = torch_fallback
        self._session: "ort.InferenceSession | None" = None

    def load(self) -> None:
        import onnxruntime as ort

        logger.info("Loading MIGraphX-backed ONNX session from %s", self.onnx_path)
        provider_options = {
            PRECISION_TO_MIGRAPHX_FLAG[self.precision]: True,
            "migraphx_exhaustive_tune": True,
        }
        self._session = ort.InferenceSession(
            str(self.onnx_path), providers=[("MIGraphXExecutionProvider", provider_options)]
        )
        if "MIGraphXExecutionProvider" not in self._session.get_providers():
            raise RuntimeError(
                f"onnxruntime did not select MIGraphXExecutionProvider for {self.onnx_path} "
                f"(got {self._session.get_providers()})"
            )

    def unload(self) -> None:
        self._session = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run one forward pass. Falls back to `torch_fallback(**inputs)` on any MIGraphX failure."""

        def _migraphx_infer() -> dict[str, torch.Tensor]:
            return self._infer_migraphx(inputs)

        def _torch_infer() -> dict[str, torch.Tensor]:
            if self.torch_fallback is None:
                raise RuntimeError(
                    f"MIGraphX execution failed for {self.onnx_path} and no torch_fallback was configured"
                )
            output = self.torch_fallback(**inputs)
            return output if isinstance(output, dict) else {"output": output}

        return run_with_fallback(f"migraphx_engine[{self.onnx_path.name}]", _migraphx_infer, _torch_infer)

    def _infer_migraphx(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._session is None:
            raise RuntimeError("Engine not loaded; call .load() before .infer()")
        import onnxruntime as ort

        # Zero-copy GPU tensor exchange via dlpack rather than a raw buffer_ptr + numpy-dtype-name
        # bind (`IOBinding.bind_input`'s traditional form) -- numpy has no native bfloat16 dtype,
        # which this project's DiT export always uses (see export/exporters/dit.py), so a
        # numpy-dtype-keyed bind would need a separate bf16 special case anyway. `OrtValue`'s
        # dlpack constructors are the documented, dtype-agnostic path for passing a live PyTorch
        # GPU tensor into onnxruntime without a host round-trip. Unverified against the installed
        # onnxruntime-rocm version's exact API surface (no ROCm hardware available in this
        # environment, per project rule) -- see docs/rocm_setup.md if this needs adjusting.
        io_binding = self._session.io_binding()
        bound_inputs = []
        for name, tensor in inputs.items():
            tensor = tensor.to(self.device).contiguous()
            bound_inputs.append(tensor)  # keep alive until run_with_iobinding returns
            ort_value = ort.OrtValue.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
            io_binding.bind_ortvalue_input(name, ort_value)
        for output_meta in self._session.get_outputs():
            io_binding.bind_output(output_meta.name, device_type=self.device.type, device_id=self.device.index or 0)

        self._session.run_with_iobinding(io_binding)

        outputs: dict[str, torch.Tensor] = {}
        for name, ort_value in zip(
            (meta.name for meta in self._session.get_outputs()), io_binding.get_outputs()
        ):
            outputs[name] = torch.utils.dlpack.from_dlpack(ort_value.to_dlpack())
        return outputs
