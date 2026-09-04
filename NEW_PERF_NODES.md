# Build a Self-Contained ComfyUI Wan Optimization Loader

I want you to create a **production-quality ComfyUI custom node** that combines model loading with optional SpargeAttention and MagCache optimization.

The implementation should be self-contained and designed specifically for ComfyUI.

## Important Architecture Requirement

I want the optimization to be applied to the returned ComfyUI `MODEL` object rather than modifying or permanently rewriting the checkpoint weights on disk.

In other words:

```text
Load diffusion model
        ↓
optional SpargeAttention patch
        ↓
optional MagCache patch
        ↓
MODEL
```

The resulting `MODEL` must remain compatible with normal ComfyUI operations, especially:

* LoRA loading
* other model patches
* samplers
* CFG
* Wan 2.2 workflows
* model cloning
* repeated generations

Do NOT create a separate sampler.

Do NOT create a separate inference pipeline.

Do NOT modify the original checkpoint files.

Do NOT require users to manually install dependencies.

---

# NODE 1 — TensorRTDiffusionLoader

Create a ComfyUI node named:

```python
TensorRTDiffusionLoader
```

It should have exactly:

### Inputs

#### 1. diffusion_model

A dropdown containing the available diffusion models found in ComfyUI's normal diffusion-model directory.

It should discover the files using ComfyUI's normal folder/path APIs rather than hard-coding paths.

For example:

```text
models/diffusion_models/
```

The dropdown should automatically refresh/discover available models.

The user selects something like:

```text
wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
```

or another compatible diffusion model.

---

#### 2. SpargeAttention

This should be a configurable SpargeAttention input.

I want the UI to make it easy to select a version/backend/configuration.

For example:

```text
SpargeAttention:
    Disabled
    Auto
    SpargeAttention 2
    SpargeAttention 2++
```

If the actual SpargeAttention package requires different configuration parameters rather than versions, expose the relevant configuration instead.

The important requirement is:

**The user should not have to manually install SpargeAttention.**

If SpargeAttention is enabled:

1. Detect whether the required package is installed.
2. If not installed, automatically install the appropriate dependency/package.
3. Detect CUDA/GPU compatibility.
4. Load the appropriate SpargeAttention implementation.
5. Patch the loaded ComfyUI model so its attention operations use SpargeAttention.
6. Return the patched model.

Do not globally monkey-patch PyTorch in a way that affects unrelated ComfyUI nodes.

Prefer model-local/module-local patching.

If a temporary patch is unavoidable, make it safely scoped to the model forward pass and restore the original implementation afterward.

The implementation must preserve compatibility with LoRAs.

---

#### 3. MagCache

Create a MagCache configuration input.

At minimum it must allow:

```text
Enabled: true/false
```

If MagCache has meaningful configuration parameters for Wan 2.2, expose them.

Prefer a UI such as:

```text
MagCache:
    Disabled
    Balanced
    Quality
    Fast
    Custom
```

For Custom, expose the relevant MagCache parameters.

The implementation must specifically support **Wan 2.2**.

Do not implement a fake generic cache.

Use the actual MagCache algorithm/implementation appropriate for Wan 2.2.

If the official MagCache implementation has Wan 2.2-specific code, use/port that logic rather than inventing a simplified approximation.

If the dependency is not installed:

1. Automatically install it.
2. Import it.
3. Load the Wan 2.2 MagCache implementation.
4. Patch the returned ComfyUI MODEL.
5. Ensure cache state is reset correctly between independent generations.

The cache must never leak state from one generation into another.

---

# NODE 1 OUTPUT

Exactly one output:

```python
("MODEL",)
```

The output should be a normal ComfyUI `MODEL`.

Example:

```text
TensorRTDiffusionLoader
        │
        ▼
      MODEL
```

This MODEL should be usable by:

```text
Load LoRA
KSampler
other MODEL patches
```

etc.

---

# NODE 2 — TensorRTCheckpointLoader

Duplicate the functionality of `TensorRTDiffusionLoader`, but create:

```python
TensorRTCheckpointLoader
```

This node should behave similarly to ComfyUI's normal checkpoint loader.

## Inputs

### 1. checkpoint

A dropdown containing checkpoints from ComfyUI's normal checkpoint directory.

Use ComfyUI's normal folder/path discovery mechanisms.

For example:

```text
models/checkpoints/
```

The user selects:

```text
some_checkpoint.safetensors
```

### 2. SpargeAttention

Same functionality as `TensorRTDiffusionLoader`.

### 3. MagCache

Same functionality as `TensorRTDiffusionLoader`.

---

# NODE 2 OUTPUT

The node must return exactly:

```python
("MODEL", "CLIP", "VAE")
```

matching the behavior expected from a normal ComfyUI checkpoint loader.

Example:

```text
TensorRTCheckpointLoader
          │
          ├── MODEL
          ├── CLIP
          └── VAE
```

The returned MODEL must already have the requested SpargeAttention and/or MagCache optimizations applied.

The CLIP and VAE should remain normal ComfyUI objects.

---

# DEPENDENCY MANAGEMENT

The entire custom node must be designed so a user can install it into:

```text
ComfyUI/custom_nodes/
```

and use it without manually installing optimization packages.

For example:

```text
ComfyUI/
└── custom_nodes/
    └── TensorRT/
        ├── __init__.py
        └── ...
```

When the node is used:

```text
if SpargeAttention enabled:
    check dependency
    install if missing

if MagCache enabled:
    check dependency
    install if missing
```

Do NOT install unnecessary dependencies when the corresponding optimization is disabled.

Do NOT run expensive compilation during every ComfyUI startup.

Dependencies should be installed lazily when first required.

Cache compiled CUDA extensions where appropriate.

Use:

```python
sys.executable
```

when invoking pip so that dependencies are installed into the same Python environment running ComfyUI.

Provide clear console messages such as:

```text
[TensorRT] SpargeAttention enabled
[TensorRT] SpargeAttention not installed
[TensorRT] Installing SpargeAttention...
[TensorRT] SpargeAttention loaded successfully
```

and:

```text
[TensorRT] MagCache enabled
[TensorRT] Loading Wan 2.2 MagCache implementation
[TensorRT] MagCache initialized
```

If installation fails, provide a useful error message explaining exactly what failed.

---

# GPU / CUDA COMPATIBILITY

Detect:

```python
torch.cuda.is_available()
torch.version.cuda
torch.cuda.get_device_name()
```

before attempting to initialize SpargeAttention.

SpargeAttention may have specific CUDA/PyTorch/GPU requirements.

Do not blindly install a package and crash later.

If the current GPU is incompatible, gracefully disable the optimization or raise a clear ComfyUI error explaining:

* detected GPU
* detected CUDA version
* detected PyTorch version
* required compatibility
* what the user should do

The implementation should be especially suitable for:

```text
NVIDIA Blackwell
RTX PRO 6000 Blackwell
```

but should not unnecessarily prevent use on compatible NVIDIA GPUs.

---

# MODEL PATCHING

This is extremely important.

Do NOT permanently alter:

```text
model weights
checkpoint files
safetensors files
```

Instead return a patched ComfyUI MODEL.

Use ComfyUI's model patching/cloning mechanisms where appropriate.

The architecture should conceptually be:

```python
model = load_model()

if sparge_enabled:
    model = apply_sparge_patch(model)

if magcache_enabled:
    model = apply_magcache_patch(model)

return model
```

The original model should remain reusable.

If the user loads the same model twice with different settings, the implementations must not interfere with one another.

For example:

```text
TensorRTDiffusionLoader
    Wan
    Sparge ON
    MagCache OFF

        ↓

MODEL A


TensorRTDiffusionLoader
    Wan
    Sparge OFF
    MagCache ON

        ↓

MODEL B
```

MODEL A and MODEL B must remain independent.

---

# LORA COMPATIBILITY

This is critical.

The returned MODEL must still support:

```text
TensorRTDiffusionLoader
        ↓
Load LoRA
        ↓
KSampler
```

and:

```text
TensorRTDiffusionLoader
        ↓
Load LoRA
        ↓
Load another LoRA
        ↓
KSampler
```

Do not bake SpargeAttention or MagCache into the weights in a way that prevents LoRA patches.

The optimization should operate on the actual forward/inference path.

---

# WAN 2.2 SUPPORT

The primary target is:

```text
Wan 2.2 14B
```

including the normal ComfyUI Wan 2.2 diffusion-model architecture.

The implementation should detect whether the loaded model appears to be Wan.

If it is not a supported architecture:

* Sparge should either safely do nothing or use a generic supported path.
* MagCache should clearly reject unsupported architectures.

Do NOT blindly apply Wan-specific MagCache logic to arbitrary diffusion models.

---

# MAGCACHE STATE MANAGEMENT

This deserves special attention.

MagCache maintains state between diffusion timesteps.

The implementation must detect a new generation/sample sequence and reset its cache.

For example:

```text
Generation 1
step 0
step 1
step 2
...
step N
       ↓
RESET CACHE
       ↓
Generation 2
step 0
step 1
...
```

Never reuse the previous generation's cached activations for a new prompt/latent/noise sequence.

Make the cache state model-local and generation-safe.

Also ensure that:

```text
model.clone()
```

does not accidentally cause multiple models to share mutable MagCache state.

---

# SPARGE + MAGCACHE COMBINATION

The two optimizations must be composable.

This should work:

```text
TensorRTDiffusionLoader
        │
        │ Sparge = ON
        │ MagCache = ON
        ▼
      MODEL
        │
        ▼
     Load LoRA
        │
        ▼
      KSampler
```

SpargeAttention should optimize attention operations.

MagCache should optimize redundant diffusion computation.

They should not interfere with each other.

---

# UI REQUIREMENTS

The nodes should be clean and simple.

For `TensorRTDiffusionLoader`:

```text
┌──────────────────────────────────────┐
│ TensorRT Diffusion Loader               │
├──────────────────────────────────────┤
│ diffusion_model: [wan2.2_14B ▼]      │
│                                      │
│ SpargeAttention: [Disabled ▼]       │
│                                      │
│ MagCache:        [Disabled ▼]       │
└──────────────────────────────────────┘
```

When custom configuration is selected, expose the additional parameters.

Avoid creating a huge wall of configuration options unless they are actually required.

Use sensible presets.

---

# ERROR HANDLING

Errors should be clear and actionable.

For example:

```text
[TensorRT] ERROR: MagCache is enabled but this model is not a supported Wan 2.2 model.
```

or:

```text
[TensorRT] ERROR: SpargeAttention could not be installed.

GPU: NVIDIA RTX PRO 6000 Blackwell
CUDA: 12.x
PyTorch: 2.x

Installation error:
...
```

Do not silently fall back to normal attention when the user explicitly enabled an optimization unless there is a UI/status indication that the optimization was unavailable.

---

# STATUS / LOGGING

When the model is loaded, print something like:

```text
[TensorRT]
Model: wan2.2_t2v_high_noise_14B_fp8_scaled
SpargeAttention: ENABLED
MagCache: ENABLED
Device: NVIDIA RTX PRO 6000 Blackwell
CUDA: 12.x
```

This should make troubleshooting easy.

---

# FILE STRUCTURE

Provide the complete implementation.

Prefer:

```text
TensorRT/
├── __init__.py
├── nodes.py
├── dependency_manager.py
├── sparge_backend.py
├── magcache_backend.py
├── model_patching.py
├── requirements.txt
└── README.md
```

However, if the implementation can genuinely be made reliable as a single self-contained class/file, that is acceptable.

The node classes themselves should remain easy to understand.

---

# IMPORTANT: VERIFY THE ACTUAL APIs

Before writing the final implementation, inspect the current official implementations/documentation for:

1. SpargeAttention
2. MagCache
3. Wan 2.2
4. ComfyUI model loading
5. ComfyUI MODEL patching

Do not invent APIs.

In particular, verify:

* current SpargeAttention package/import names
* current SpargeAttention function signatures
* current CUDA requirements
* current MagCache Wan 2.2 implementation
* current Wan 2.2 model structure
* correct ComfyUI model patching mechanism
* correct ComfyUI folder/path discovery APIs

Use the actual current source code/API rather than relying on memory.

---

# DELIVERABLE

Return:

1. Complete source code for every required file.
2. Exact folder structure.
3. Exact installation instructions.
4. Explanation of how SpargeAttention is patched.
5. Explanation of how MagCache is patched.
6. Explanation of how cache state is reset.
7. Explanation of how LoRA compatibility is preserved.
8. Explanation of automatic dependency installation.
9. Any CUDA/PyTorch compatibility caveats.
10. A simple example ComfyUI workflow showing:

```text
TensorRTDiffusionLoader
        ↓
Load LoRA
        ↓
KSampler
```

and another showing:

```text
TensorRTDiffusionLoader
        ↓
Load LoRA
        ↓
KSampler
```

with both SpargeAttention and MagCache enabled.

Most importantly, **give me working code based on the current official APIs, not pseudocode.**
