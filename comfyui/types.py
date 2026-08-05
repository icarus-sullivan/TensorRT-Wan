"""ComfyUI socket type names used between TensorRT-Wan nodes.

Nodes that have a direct ComfyUI equivalent reuse ComfyUI's own type strings (`"LATENT"`,
`"IMAGE"`, `"CONDITIONING"`) so TensorRT-Wan nodes can be dropped into existing Wan workflows and
connect straight into non-TensorRT nodes (a stock VAEDecode, SaveImage, etc.) — see
docs/comfyui_integration.md. Types with no ComfyUI equivalent (a loaded runtime, a built engine
handle) get their own `TRTWAN_*` socket type so mismatched connections are caught by ComfyUI's
own graph validation instead of failing at run time.
"""

RUNTIME = "TRTWAN_RUNTIME"
DIT_ENGINE = "TRTWAN_DIT_ENGINE"
TEXT_ENCODER_ENGINE = "TRTWAN_TEXT_ENCODER_ENGINE"
VAE_ENCODER_ENGINE = "TRTWAN_VAE_ENCODER_ENGINE"
VAE_DECODER_ENGINE = "TRTWAN_VAE_DECODER_ENGINE"
CONDITIONING_MANAGER = "TRTWAN_CONDITIONING_MANAGER"
UNIFIED_CONDITIONING = "TRTWAN_UNIFIED_CONDITIONING"
# Payload is a tensorrt_wan.conditioning.types.ConditioningTensor — already encoded by whichever
# node produced it (Text Encoder, VAE Encoder). The Conditioning Manager node only merges these
# (via ConditioningManager.combine_encoded), it does not re-run encoding.
COND_INPUT = "TRTWAN_COND_INPUT"
SCHEDULER = "TRTWAN_SCHEDULER"
MODEL_CONFIG = "TRTWAN_MODEL_CONFIG"
