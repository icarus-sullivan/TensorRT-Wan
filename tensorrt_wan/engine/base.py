"""Generic TensorRT engine wrapper: deserialize once, bind tensors, execute_async_v3, sync.

Every concrete engine (text encoder, DiT, VAE encoder/decoder) composes one of these rather than
touching the TensorRT API directly — this is the one place binding setup, stream management, and
fallback wiring live, so a TensorRT API change (e.g. a future `execute_async_v4`) is a one-file fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from tensorrt_wan.runtime.fallback import run_with_fallback
from tensorrt_wan.utils.logging import get_logger

if TYPE_CHECKING:
    import tensorrt as trt

logger = get_logger(__name__)


class TensorRTEngineWrapper:
    """Loads a serialized `.engine` file and runs it against named input/output tensors.

    `torch_fallback`, if given, is called via `runtime.fallback.run_with_fallback` whenever the
    TensorRT path raises — per the project's automatic-fallback rule, a plugin/shape/precision
    failure degrades to eager PyTorch instead of crashing generation.
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | None = None,
        torch_fallback: "torch.nn.Module | None" = None,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.device = device or torch.device("cuda")
        self.torch_fallback = torch_fallback
        self._engine: "trt.ICudaEngine | None" = None
        self._context: "trt.IExecutionContext | None" = None
        self._stream: torch.cuda.Stream | None = None

    def load(self) -> None:
        """Deserialize `engine_path` and create an execution context + dedicated CUDA stream."""
        import tensorrt as trt

        logger.info("Loading TensorRT engine from %s", self.engine_path)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        engine_bytes = self.engine_path.read_bytes()
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine at {self.engine_path}")
        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=self.device)

    def unload(self) -> None:
        self._context = None
        self._engine = None
        self._stream = None

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run one forward pass. Falls back to `torch_fallback(**inputs)` on any TensorRT failure."""

        def _trt_infer() -> dict[str, torch.Tensor]:
            return self._infer_trt(inputs)

        def _torch_infer() -> dict[str, torch.Tensor]:
            if self.torch_fallback is None:
                raise RuntimeError(
                    f"TensorRT execution failed for {self.engine_path} and no torch_fallback was configured"
                )
            output = self.torch_fallback(**inputs)
            return output if isinstance(output, dict) else {"output": output}

        return run_with_fallback(f"engine[{self.engine_path.name}]", _trt_infer, _torch_infer)

    def _infer_trt(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._context is None or self._engine is None or self._stream is None:
            raise RuntimeError("Engine not loaded; call .load() before .infer()")

        context, engine, stream = self._context, self._engine, self._stream
        outputs: dict[str, torch.Tensor] = {}

        with torch.cuda.stream(stream):
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if name in inputs:
                    # set_tensor_address hands TensorRT a raw pointer with no dtype conversion —
                    # if the caller's tensor dtype doesn't match what this input was built with,
                    # TensorRT reinterprets the same bytes as its own dtype rather than erroring.
                    # Confirmed as a real bug via a real generation run: FlowMatchEulerScheduler's
                    # `timestep` is float32 (torch.linspace's default), but the DiT engine's
                    # `timestep` input was exported as float16 — every element silently became
                    # byte-garbage, producing NaN on the very first denoising step. Casting to the
                    # engine's own declared dtype here fixes this generically for every input on
                    # every engine, not just this one case. See docs/wan2.2_i2v_14b_notes.md.
                    target_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
                    tensor = inputs[name].to(self.device, dtype=target_dtype, non_blocking=True).contiguous()
                    # set_input_shape returns a bool rather than raising on failure (e.g. a rank
                    # mismatch against what the engine expects) — confirmed via a real run that
                    # silently ignoring that return value lets execution continue with a
                    # stale/wrong shape binding instead of failing loudly. See
                    # docs/wan2.2_i2v_14b_notes.md.
                    if not context.set_input_shape(name, tuple(tensor.shape)):
                        raise RuntimeError(
                            f"set_input_shape failed for input {name!r} with shape {tuple(tensor.shape)} "
                            f"against engine {self.engine_path} — likely a rank or bounds mismatch "
                            "against the profile this engine was built with."
                        )
                    context.set_tensor_address(name, tensor.data_ptr())
                    inputs[name] = tensor  # keep the contiguous copy alive until execution completes
                else:
                    shape = tuple(context.get_tensor_shape(name))
                    dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
                    out = torch.empty(shape, dtype=dtype, device=self.device)
                    context.set_tensor_address(name, out.data_ptr())
                    outputs[name] = out

            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outputs


def _trt_dtype_to_torch(dtype: "trt.DataType") -> torch.dtype:
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.FP8: torch.float8_e4m3fn,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]
