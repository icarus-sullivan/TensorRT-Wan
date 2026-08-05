"""Stage 2: torch.export.ExportedProgram -> ONNX.

Uses `torch.onnx.export(..., dynamo=True)` against the already-exported program rather than
re-tracing the raw `nn.Module`, so the dynamic shape ranges established in
`export.torch_export` carry through unchanged instead of being re-derived by the ONNX exporter.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tensorrt_wan.export.base import ModelExporter
from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    exported_program: torch.export.ExportedProgram,
    exporter: ModelExporter,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("ONNX export: %s -> %s (opset=%d)", exporter.name, output_path, exporter.opset_version)
    onnx_program = torch.onnx.export(
        exported_program,
        (),
        kwargs=exporter.example_inputs(),
        input_names=exporter.input_names,
        output_names=exporter.output_names,
        opset_version=exporter.opset_version,
        dynamo=True,
    )
    onnx_program.save(str(output_path))
    return output_path
