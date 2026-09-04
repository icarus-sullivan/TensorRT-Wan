#!/usr/bin/env python3
"""Force-build (and cache) the TensorRT-RT Wan VAE encoder/decoder engines without running a full
ComfyUI workflow.

`vae_rt.py`'s nodes build engines lazily on first real use inside a workflow -- normally fine, but
it means the (slow) first build happens mid-generation, and a build failure gets silently masked by
the eager-PyTorch fallback. This script calls the same build path directly and, unlike the node,
lets a build failure raise instead of falling back -- so you get a clear pass/fail signal.

Usage (run on the pod, in the same Python environment ComfyUI itself uses):

    python build_vae_engine.py
    python build_vae_engine.py --encoder-frames 1 --latent-frames 21
    python build_vae_engine.py --vae-name wan2.2_vae.safetensors --precision fp32
    python build_vae_engine.py --skip-encoder --no-chunk   # reproduce the known decoder OOM directly

No resolution flags: the built engine already covers the *entire* dynamic-shape range declared by
vae_rt.py's own ENCODER_HEIGHT/WIDTH and DECODER_LATENT_HEIGHT/WIDTH constants (currently
256-1536px / 32-192 latent) -- that range is what's actually dynamic at runtime, arbitrary
width/height within it needs no rebuild. This script traces the export using each profile's own
"opt" shape as the example (torch.export needs *some* concrete shape even for a Dim.AUTO axis,
but which one barely matters -- it doesn't narrow what the built engine accepts). Only
--encoder-frames/--latent-frames are real per-engine choices: frame count genuinely can't be made
dynamic (see vae_rt.py's module docstring), so a different frame count is a different cached
engine, not a wider profile.

If --latent-frames exceeds vae_rt.DECODER_CHUNK_FRAMES (9 by default), the decoder build goes
through the experimental chunked path (vae_rt.py's `_decode_chunked_trt`) instead of one
full-length build -- a full-length decoder build at real target frame counts (e.g. 21) is a
confirmed ~100GB TensorRT/Myelin ForeignNode OOM on this project's GPU (Blackwell sm_120), not
something this script can avoid on your behalf. --no-chunk forces the old single-shot path if you
want to reproduce that directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _load_vae_rt():
    node_dir = Path(__file__).resolve().parent / "nodes"
    spec = importlib.util.spec_from_file_location("vae_rt", node_dir / "vae_rt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui-root", default=os.environ.get("COMFYUI_ROOT", "/workspace/runpod-slim/ComfyUI"))
    parser.add_argument("--vae-name", default=None, help="default: vae_rt.DEFAULT_VAE_FILENAME")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default=None, help="default: vae_rt.DEFAULT_PRECISION")
    parser.add_argument("--encoder-frames", type=int, default=1, help="T for the encoder engine (1 = single reference image)")
    parser.add_argument("--latent-frames", type=int, default=21, help="T for the decoder engine")
    parser.add_argument("--skip-encoder", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument(
        "--no-chunk", action="store_true",
        help="force a single full-length decoder build instead of the chunked path (to reproduce/compare "
        "against the known ~100GB Myelin ForeignNode OOM at latent-frames > DECODER_CHUNK_FRAMES)",
    )
    args = parser.parse_args()

    os.environ["COMFYUI_ROOT"] = args.comfyui_root
    if args.comfyui_root not in sys.path:
        sys.path.insert(0, args.comfyui_root)

    vae_rt = _load_vae_rt()
    import torch

    vae_name = args.vae_name or vae_rt.DEFAULT_VAE_FILENAME
    precision = args.precision or vae_rt.DEFAULT_PRECISION

    print(f"Resolving checkpoint {vae_name!r} (downloads from HuggingFace if not already in models/vae/)...")
    checkpoint_path = vae_rt._ensure_vae_checkpoint(vae_name)
    print(f"  -> {checkpoint_path}")

    runtime = vae_rt._VAERuntime(checkpoint_path, precision=precision)
    device, dtype = runtime.device, runtime.dtype

    if not args.skip_encoder:
        h_min, h_opt, h_max = vae_rt.ENCODER_HEIGHT
        w_min, w_opt, w_max = vae_rt.ENCODER_WIDTH
        print(
            f"\nBuilding vae_encoder engine: frames={args.encoder_frames}, {precision}, "
            f"covers {h_min}-{h_max} x {w_min}-{w_max}px (tracing at opt {h_opt}x{w_opt})..."
        )
        pixels = torch.zeros(1, 3, args.encoder_frames, h_opt, w_opt, device=device, dtype=dtype)
        runtime._get_or_build(
            "vae_encoder", vae_rt._VAEEncodeWrapper, "pixels", "latent", pixels, {3: None, 4: None},
            (
                (1, 3, args.encoder_frames, h_min, w_min),
                (1, 3, args.encoder_frames, h_opt, w_opt),
                (1, 3, args.encoder_frames, h_max, w_max),
            ),
            args.encoder_frames,
        )
        print("  -> vae_encoder engine built and cached.")

    if not args.skip_decoder:
        channels = vae_rt.VAE_SOURCES.get(vae_name, {}).get("latent_channels", 16)
        h_opt, w_opt = vae_rt.DECODER_LATENT_HEIGHT[1], vae_rt.DECODER_LATENT_WIDTH[1]
        latent = torch.zeros(1, channels, args.latent_frames, h_opt, w_opt, device=device, dtype=dtype)

        # Calling the real runtime methods (not duplicating profile-shape construction here) so
        # this script exercises exactly the same code path `TensorRTWanVAEDecode` does -- except
        # it calls _decode_chunk_trt/_decode_chunked_trt directly instead of the public decode(),
        # so a build failure raises here instead of being silently masked by decode()'s own
        # eager-PyTorch fallback. That fallback is the whole reason this script exists.
        if args.latent_frames > vae_rt.DECODER_CHUNK_FRAMES and not args.no_chunk:
            print(
                f"\nBuilding vae_decoder via chunked decode: frames={args.latent_frames} > "
                f"DECODER_CHUNK_FRAMES={vae_rt.DECODER_CHUNK_FRAMES}, {precision} -- builds one engine "
                f"per distinct chunk size (<= {vae_rt.DECODER_CHUNK_FRAMES} frames, "
                f"{vae_rt.DECODER_CHUNK_OVERLAP}-frame overlap) instead of one "
                f"T={args.latent_frames} engine, which is the known ~100GB OOM (see --no-chunk to "
                "reproduce it directly)..."
            )
            runtime._decode_chunked_trt(latent)
            print("  -> vae_decoder chunk engine(s) built and cached.")
        else:
            print(f"\nBuilding vae_decoder engine: frames={args.latent_frames}, {precision}...")
            runtime._decode_chunk_trt(latent)
            print("  -> vae_decoder engine built and cached.")

    print(f"\nDone. Cached under {runtime._cache_dir()}")


if __name__ == "__main__":
    main()
