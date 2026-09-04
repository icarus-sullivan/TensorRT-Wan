from __future__ import annotations

import argparse
import json
import os

from tensorrt_wan.cli.loader import resolve_loader
from tensorrt_wan.export.exporters import DiTExporter, TextEncoderExporter, VAEDecoderExporter, VAEEncoderExporter
from tensorrt_wan.export.onnx_export import export_to_onnx
from tensorrt_wan.export.torch_export import export_to_torch_export

_EXPORTERS = {
    "text_encoder": TextEncoderExporter,
    "dit": DiTExporter,
    "vae_encoder": VAEEncoderExporter,
    "vae_decoder": VAEDecoderExporter,
}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export", help="PyTorch -> ONNX export")
    export_sub = parser.add_subparsers(dest="export_command", required=True)

    onnx_parser = export_sub.add_parser("onnx", help="Export a Wan submodule to ONNX")
    onnx_parser.add_argument("--component", choices=sorted(_EXPORTERS), required=True)
    onnx_parser.add_argument(
        "--loader",
        required=True,
        help="'module.path:function_name' returning a loaded nn.Module given --checkpoint",
    )
    onnx_parser.add_argument("--checkpoint", required=True, help="Path passed to the loader function")
    onnx_parser.add_argument("--output", required=True, help="Output .onnx path")
    onnx_parser.add_argument(
        "--exporter-kwargs",
        default="{}",
        help="JSON dict of exporter-specific dims, e.g. '{\"in_channels\": 36, \"text_dim\": 4096}' for dit",
    )
    onnx_parser.add_argument(
        "--target",
        choices=["tensorrt", "migraphx"],
        default="tensorrt",
        help=(
            "Downstream builder this export is for. 'migraphx' (AMD/ROCm, no TensorRT -- see "
            "docs/rocm_setup.md) decomposes RMSNorm and caps the ONNX opset at 19, since "
            "MIGraphX has no native RMSNormalization/Attention op support above that -- see "
            "examples/loaders/wan_comfyui_loader.py's load_dit() and export/base.py's "
            "opset_version docstrings for why."
        ),
    )
    onnx_parser.set_defaults(func=run_onnx)


def run_onnx(args: argparse.Namespace) -> int:
    # Read by examples/loaders/wan_comfyui_loader.py's load_dit() (RMSNorm decomposition) and
    # export/base.py's ModelExporter.opset_version (opset cap) -- must be set before either the
    # loader or the exporter's opset_version property run, so it's set here first, once.
    os.environ["TRTWAN_EXPORT_TARGET"] = args.target

    loader = resolve_loader(args.loader)
    model = loader(args.checkpoint)

    exporter_cls = _EXPORTERS[args.component]
    exporter_kwargs = json.loads(args.exporter_kwargs)
    exporter = exporter_cls(model, **exporter_kwargs)

    exported_program = export_to_torch_export(exporter)
    output_path = export_to_onnx(exported_program, exporter, args.output)
    print(f"Exported {args.component} -> {output_path}")
    return 0
