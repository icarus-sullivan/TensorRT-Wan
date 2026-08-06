"""--loader adapter for `trtwan export onnx` / `trtwan build engine` that reuses ComfyUI's own
Wan model-loading code instead of reimplementing Wan's architecture in this repo.

See docs/wan2.2_i2v_14b_notes.md for how this was derived (traced from `UNETLoader.load_unet` in
ComfyUI's nodes.py) and verified end to end on RunPod hardware: `load_dit()` here, plus a
standalone `torch.export.export()` call against the real 14.29B-param checkpoint, both succeed
(see the notes doc's "torch.export succeeded" section).

Usage:
    COMFYUI_ROOT=/workspace/runpod-slim/ComfyUI trtwan export onnx \
        --component dit \
        --loader examples.loaders.wan_comfyui_loader:load_dit \
        --checkpoint /workspace/runpod-slim/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
        --output dit_high_noise.onnx \
        --exporter-kwargs '{"in_channels": 36, "text_dim": 4096}'

`in_channels=36`, not 16 — see docs/wan2.2_i2v_14b_notes.md's conditioning-mismatch section.
`DiTExporter`/`DiTEngine` now use the right input names (`x`/`timestep`/`context`, confirmed
against `WanModel.forward`) and the right shape, but still don't *construct* a real 36-channel
`x` for I2V (noise + image-latent + mask, channel-concatenated) — `example_inputs()` traces
against zeros. Fine for a T2V-only export attempt; a real I2V export still needs that built
first. `DiTEngine._build_inputs` also raises `NotImplementedError` if any non-text conditioning
is present, rather than silently mis-routing it, for the same reason.
"""

from __future__ import annotations

import os
import sys

import torch

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch RoPE application. Cloned (not imported) from comfy/ldm/flux/math.py's own
    `_apply_rope1` fallback, monkeypatched into comfy.ldm.wan.model in place of the default
    `comfy_kitchen` custom op — see the comment at the call site in `load_dit` for why.

    Cloned rather than depended on: (1) this loader should keep working even outside a ComfyUI
    environment eventually — a from-scratch/non-ComfyUI loader needs this exact math anyway, so
    it belongs owned in this project, not borrowed at runtime from comfy internals that could
    change; (2) it decouples us from `comfy.model_management.in_training`'s exact semantics and
    blast radius (that flag may gate other comfy_kitchen ops we haven't exercised/verified) —
    patching this one function is scoped to exactly what's been confirmed necessary.

    Correctness note: this is Wan's actual convention — interleaved-pair 2D rotation via a
    per-position 2x2 matrix from `rope()` — NOT the rotate-half convention our own
    `RotaryEmbedding` TensorRT plugin (`plugins/csrc/rotary_embedding/`) currently implements.
    See docs/wan2.2_i2v_14b_notes.md's RoPE-convention finding; the plugin needs fixing to match
    this before it can be trusted. This clone is the reference to fix it against.
    """
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, : x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


def _decomposed_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Reference (matmul + softmax + matmul) scaled dot-product attention — the same decomposition
    PyTorch's own docs give as `scaled_dot_product_attention`'s defining math.

    Confirmed necessary against real builds: `torch.onnx`'s dynamo exporter maps
    `F.scaled_dot_product_attention` straight to ONNX opset 23's native `Attention` op, and
    TensorRT 11.2's importer for that native op fails with `MyelinCheckException: ... Attention
    operation was not supported by a dedicated kernel` for both the text encoder's masked
    self-attention and the VAE's (unmasked) bottleneck self-attention — i.e. this isn't
    mask-specific, the native-op import path itself doesn't have a fused kernel available for
    either shape in this TensorRT version. The suggested fix in the error text
    (`IAttention::setDecomposable`) isn't reachable from Python in this TensorRT version either
    (confirmed separately: `network.get_layer()` returns the generic `ILayer` for
    `ATTENTION_INPUT`/`ATTENTION_OUTPUT` layers, no downcast, no Python constructor). Decomposing
    before export instead sidesteps the native op entirely — the resulting ONNX graph is plain
    MatMul/Softmax nodes, which TensorRT's older, doc-confirmed MHA fusion pass can still
    recognize and fuse (see docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/
    transformers-fused-attention.html). See docs/wan2.2_i2v_14b_notes.md's 2026-08-06 session
    section for the full investigation.
    """
    if enable_gqa:
        n_rep = query.shape[-3] // key.shape[-3]
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=-3)
            value = value.repeat_interleave(n_rep, dim=-3)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    attn_weight = query @ key.transpose(-2, -1) * scale
    if is_causal:
        seq_q, seq_k = query.shape[-2], key.shape[-2]
        causal_mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=query.device).tril()
        attn_weight = attn_weight.masked_fill(~causal_mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def _decompose_attention_for_export() -> None:
    """Monkeypatch `torch.nn.functional.scaled_dot_product_attention` to `_decomposed_sdpa` for
    the duration of this process. Every attention call path in this loader (`comfy.ops`'s
    `scaled_dot_product_attention` wrapper, used by both the text encoder and the VAE) bottoms
    out in the real `torch.nn.functional.scaled_dot_product_attention` regardless of which
    internal branch it takes, so patching that one global function covers both — see
    `_decomposed_sdpa`'s docstring for why this is necessary. Only affects this process, not a
    running ComfyUI server (separate process/memory space), same reasoning as the `apply_rope1`
    monkeypatch in `load_dit`. Not applied in `load_dit` itself: the DiT's own attention already
    finds a dedicated fused kernel with no such error, so it doesn't need this and re-decomposing
    it would only cost performance for no correctness benefit.
    """
    torch.nn.functional.scaled_dot_product_attention = _decomposed_sdpa


def _add_comfyui_to_path() -> None:
    """ComfyUI isn't a pip-installed package — its `comfy` module only imports if ComfyUI's own
    repo root is on sys.path. Set COMFYUI_ROOT, or this defaults to the path this loader was
    originally written against.
    """
    comfyui_root = os.environ.get("COMFYUI_ROOT", "/workspace/runpod-slim/ComfyUI")
    if comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)


def load_dit(checkpoint_path: str) -> torch.nn.Module:
    """Load a Wan DiT checkpoint (.safetensors under ComfyUI's `diffusion_models/`) via
    `comfy.sd.load_diffusion_model` and return the underlying transformer module, moved to GPU
    and cast to the target export dtype.

    `comfy.sd.load_diffusion_model` alone is not enough: ComfyUI keeps a full-precision (fp32)
    master copy on the CPU "offload device" and only moves/casts a model when its own execution
    engine runs a forward pass, which we never trigger here — confirmed empirically on the
    RunPod instance, see docs/wan2.2_i2v_14b_notes.md's "actually ran" section. Without the
    explicit `.to()` below, `torch.export` would trace against fp32-on-CPU tensors instead of
    what the built engine is actually meant to run.

    This is the single high-noise or low-noise expert for a Wan 2.2 MoE checkpoint — there is no
    engine-level support in this repo yet for switching between two experts mid-schedule (see
    docs/wan2.2_i2v_14b_notes.md's MoE section); call this once per expert and export/build each
    as its own engine.

    Override device/dtype via `TRTWAN_LOADER_DEVICE` (default `cuda`) / `TRTWAN_LOADER_DTYPE`
    (default `fp16`; one of fp32/fp16/bf16) if you need to debug against a CPU or fp32 trace.
    """
    _add_comfyui_to_path()

    import comfy.ldm.wan.model as wan_model  # noqa: E402
    import comfy.sd  # noqa: E402

    # comfy/ldm/wan/model.py imports apply_rope1 by name from comfy/ldm/flux/math.py, which
    # dispatches to an opaque `comfy_kitchen` custom op with no ONNX translation by default —
    # confirmed to break torch.onnx.export with "No ONNX function found for
    # comfy_kitchen.apply_rope1" on real hardware (see docs/wan2.2_i2v_14b_notes.md). Monkeypatch
    # just this one call site to our own cloned pure-PyTorch implementation (`_apply_rope1`
    # above) rather than flipping comfy.model_management.in_training, a broad global flag whose
    # full effect on comfy's other custom-kernel dispatch we haven't audited. Only affects this
    # process, not a running ComfyUI server (separate process/memory space).
    wan_model.apply_rope1 = _apply_rope1

    device = os.environ.get("TRTWAN_LOADER_DEVICE", "cuda")
    dtype = _DTYPES[os.environ.get("TRTWAN_LOADER_DTYPE", "fp16")]

    model_patcher = comfy.sd.load_diffusion_model(checkpoint_path, model_options={})
    diffusion_model = model_patcher.model.diffusion_model
    diffusion_model = diffusion_model.to(device=device, dtype=dtype)

    # forward_orig (comfy/ldm/wan/model.py) does `self.patch_embedding(x.float()).to(x.dtype)` -
    # it deliberately runs this one conv in fp32 regardless of the rest of the model's dtype, then
    # casts the result straight back down. comfy.ops's conv here does no dtype reconciliation of
    # its own (unlike some of its other op variants), so patch_embedding's weight must actually be
    # fp32 to match, or eager/export both fail on a dtype mismatch. Cheap to do - this submodule is
    # ~1.3M params - versus casting the whole 14.29B-param model to fp32 (which OOM'd a 95GB GPU
    # already holding another ComfyUI-loaded copy; see docs/wan2.2_i2v_14b_notes.md).
    if dtype != torch.float32 and hasattr(diffusion_model, "patch_embedding"):
        diffusion_model.patch_embedding = diffusion_model.patch_embedding.to(dtype=torch.float32)

    diffusion_model.eval()
    return diffusion_model


class _TextEncoderWrapper(torch.nn.Module):
    """`comfy.text_encoders.t5.T5.forward` returns `(x, intermediate)` — `intermediate` is only
    populated when a caller asks for a specific `intermediate_output` layer (SD1ClipModel's
    hidden-state-extraction feature), which the TensorRT `TextEncoderExporter`/`TextEncoderEngine`
    contract has no use for (single `text_embeds` output only). This wrapper is exactly what
    `example_inputs()`'s `input_ids`/`attention_mask` names get called against; without it,
    `torch.export` would trace `exporter.model.__call__` against a 2-tuple return, mismatching
    `TextEncoderExporter.output_names`'s single `text_embeds`.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.transformer(input_ids, attention_mask)[0]


def load_text_encoder(checkpoint_path: str) -> torch.nn.Module:
    """Load Wan's UMT5-XXL text encoder via `comfy.sd.load_clip` and return the raw transformer
    (not ComfyUI's `CLIP` wrapper, which handles tokenization/chunking/weighting — none of which
    the exported engine needs; `TextEncoderEngine.encode_text` does its own tokenization on CPU
    and calls the engine directly with `input_ids`/`attention_mask`).

    Real attribute path confirmed on RunPod hardware (not documented anywhere in ComfyUI's own
    code comments): `comfy.sd.load_clip(..., clip_type=CLIPType.WAN)` returns a `CLIP` whose
    `.cond_stage_model` is a `WanTEModel` (`comfy.text_encoders.wan.te`'s closure class) with a
    `.umt5xxl` attribute (an `SDClipModel`, named after the `name="umt5xxl"` kwarg
    `WanT5Model.__init__` passes up) — `.umt5xxl.transformer` is the actual `comfy.text_encoders
    .t5.T5` module, whose `forward(input_ids, attention_mask, ...)` matches
    `TextEncoderExporter`'s input names directly (see `_TextEncoderWrapper` above for the one
    mismatch: its 2-tuple return).
    """
    _add_comfyui_to_path()
    _decompose_attention_for_export()

    import comfy.sd  # noqa: E402
    import folder_paths  # noqa: E402

    device = os.environ.get("TRTWAN_LOADER_DEVICE", "cuda")
    dtype = _DTYPES[os.environ.get("TRTWAN_LOADER_DTYPE", "fp16")]

    clip = comfy.sd.load_clip(
        ckpt_paths=[checkpoint_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.WAN,
        model_options={},
    )
    transformer = clip.cond_stage_model.umt5xxl.transformer
    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()
    return _TextEncoderWrapper(transformer)


class _VAEEncodeWrapper(torch.nn.Module):
    """`comfy.ldm.wan.vae2_2.WanVAE` has `.encode(x)`/`.decode(z)` methods, not a single unified
    `forward` — `torch.export.export(module, ...)` traces `module.__call__`/`forward`, so each
    direction needs its own thin wrapper exposing the method it needs as `forward`. This one
    exists for `VAEEncoderExporter` (`pixels` -> `latent`); see `_VAEDecodeWrapper` below for the
    other direction.
    """

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(pixels)


class _VAEDecodeWrapper(torch.nn.Module):
    """See `_VAEEncodeWrapper` above. This direction is for `VAEDecoderExporter` (`latent` ->
    `pixels`)."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent)


def _load_wan_vae(checkpoint_path: str) -> torch.nn.Module:
    """Shared by `load_vae_encoder`/`load_vae_decoder` — both directions live in the same
    checkpoint/module (`comfy.ldm.wan.vae2_2.WanVAE`, confirmed via `comfy.sd.VAE`'s dispatch on
    the real `wan2.2_vae.safetensors` file on RunPod hardware), so there's one real load here and
    two thin per-direction wrappers around it, not two separate loads.

    Caller (`export`/`build` CLI) treats `checkpoint_path` as opaque — for this VAE it must be
    ComfyUI's own `models/vae/wan2.2_vae.safetensors`, not a diffusion_models/text_encoders path.
    """
    _add_comfyui_to_path()
    _decompose_attention_for_export()

    import comfy.ops  # noqa: E402
    import comfy.sd  # noqa: E402
    import comfy.utils  # noqa: E402

    # comfy.ops.Conv3d._conv_forward calls torch.cudnn_convolution directly (bypassing
    # nn.Conv3d's normal _conv_forward/F.conv3d) whenever NVIDIA_MEMORY_CONV_BUG_WORKAROUND is
    # True — a deliberate, real workaround for an actual cuDNN memory bug (gated on cuDNN
    # 9.10.2-9.15.0 + torch 2.9-2.10, which matches this environment exactly, see comfy/ops.py).
    # `torch.cudnn_convolution` has no FakeTensor/meta kernel registered, so torch.export fails
    # with `UnsupportedOperatorException: aten.cudnn_convolution.default` — confirmed against a
    # real export attempt on RunPod hardware. Safe to disable for the duration of export only:
    # non-strict `torch.export` traces against FakeTensors, never runs a real cuDNN kernel, so
    # the memory bug this workaround exists for is simply not in play here. Only affects this
    # process, not a running ComfyUI server (separate process/memory space) — same reasoning as
    # the `apply_rope1` monkeypatch above.
    comfy.ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND = False

    device = os.environ.get("TRTWAN_LOADER_DEVICE", "cuda")
    dtype = _DTYPES[os.environ.get("TRTWAN_LOADER_DTYPE", "fp16")]

    sd, metadata = comfy.utils.load_torch_file(checkpoint_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    first_stage_model = vae.first_stage_model.to(device=device, dtype=dtype)
    first_stage_model.eval()
    return first_stage_model


def load_vae_encoder(checkpoint_path: str) -> torch.nn.Module:
    """See `_load_wan_vae`/`_VAEEncodeWrapper`."""
    return _VAEEncodeWrapper(_load_wan_vae(checkpoint_path))


def load_vae_decoder(checkpoint_path: str) -> torch.nn.Module:
    """See `_load_wan_vae`/`_VAEDecodeWrapper`."""
    return _VAEDecodeWrapper(_load_wan_vae(checkpoint_path))
