# plan.md
# TensorRT Runtime Adapter System
#
# Goal:
# Add a generic runtime adapter system to the existing TensorRT backend without
# changing the existing TensorRT execution engine. The current engine is already
# functioning for WAN 2.2 i2v and should remain the source of truth.
#
# The adapter runtime should allow dynamic LoRA loading and future support for
# other adapter types without requiring TensorRT engine rebuilds.
#
# -----------------------------------------------------------------------------

# Design Goals

- DO NOT modify the existing TensorRT engine architecture.
- DO NOT require rebuilding engines when adapters change.
- Keep all activation data on the GPU.
- Allow unlimited adapter types.
- Allow multiple LoRAs simultaneously.
- Support dynamic loading/unloading during runtime.
- Keep the design modular.
- Keep the runtime independent of ComfyUI nodes.
- Make future adapter types plug-and-play.

---

# Overall Architecture
* this overall is an example not 100% like our architecutre but the adapter runtime + lora activation can flow where needed in our own setup

```
                ComfyUI

                    │
                    ▼

        TensorRT Backend Runtime

    ┌─────────────────────────────┐
    │                             │
    │        Engine Manager        │
    │                             │
    └──────────────┬──────────────┘
                   │
                   ▼

          TensorRT Execution

                   │

          Activation Hook

                   │

                   ▼

            Adapter Runtime

      ┌────────────┼────────────┐
      ▼            ▼            ▼

   LoRA        IPAdapter    ControlNet

      ▼            ▼            ▼

     Modified GPU Activations

                   │

                   ▼

      Continue TensorRT Execution
```

The TensorRT engine should never know adapters exist.

The Adapter Runtime owns all runtime modifications.

---

# Runtime Adapter Interface

Every runtime modification derives from one common interface.

```python
class RuntimeAdapter:

    def initialize(self):
        """
        Allocate GPU resources.
        Parse model.
        Prepare runtime buffers.
        """

    def supports_layer(self, layer_id):
        """
        Return True if this adapter modifies the requested layer.
        """

    def before_layer(self, layer_id, activation):
        """
        Optional hook before TensorRT executes a layer.
        Default implementation returns activation unchanged.
        """

        return activation

    def after_layer(self, layer_id, activation):
        """
        Optional hook after TensorRT executes a layer.
        Default implementation returns activation unchanged.
        """

        return activation

    def unload(self):
        """
        Release GPU memory.
        """
```

---

# Why One Interface

The runtime should never know which adapter type it is executing.

Every adapter becomes interchangeable.

Examples:

- LoRA
- LyCORIS
- DoRA
- ControlNet
- IPAdapter
- Future adapters

All behave identically from the runtime perspective.

---

# LoRA Adapter

```python
class LoRAAdapter(RuntimeAdapter):

    def __init__(self,
                 safetensors,
                 strength):

        self.layers = {}

        self.strength = strength

    def initialize(self):

        """
        Parse safetensors.

        Convert tensors to GPU buffers.

        Build lookup table.
        """

    def supports_layer(self, layer_id):

        return layer_id in self.layers

    def after_layer(self,
                    layer_id,
                    activation):

        delta = self.compute_delta(
            layer_id,
            activation
        )

        return activation + delta
```

The adapter owns:

- parsing
- GPU memory
- LoRA weights
- delta computation

It never owns TensorRT execution.

---

# Adapter Manager

The runtime owns exactly one Adapter Manager.

Responsibilities:

- Register adapters
- Remove adapters
- Enable adapters
- Disable adapters
- Call adapters
- Maintain execution order

Pseudo implementation:

```python
class AdapterManager:

    adapters = []

    def register(adapter):

        adapter.initialize()

        adapters.append(adapter)

    def unregister(adapter):

        adapter.unload()

        adapters.remove(adapter)

    def process_before(layer,
                       activation):

        for adapter in adapters:

            if adapter.supports_layer(layer):

                activation = adapter.before_layer(
                    layer,
                    activation
                )

        return activation

    def process_after(layer,
                      activation):

        for adapter in adapters:

            if adapter.supports_layer(layer):

                activation = adapter.after_layer(
                    layer,
                    activation
                )

        return activation
```

---

# Execution Flow

Current execution should become:

```
TensorRT Layer

↓

AdapterManager.process_after()

↓

Next TensorRT Layer

↓

AdapterManager.process_after()

↓

Next Layer

↓

Repeat
```

TensorRT remains responsible for all model execution.

The Adapter Manager only modifies activations.

---

# Multiple LoRAs

The runtime must support:

- zero LoRAs
- one LoRA
- many LoRAs

The engine should never care how many exist.

Example:

```
Layer Output

↓

LoRA A

+

LoRA B

+

LoRA C

↓

Combined Delta

↓

Return Activation
```

---

# Recommended Flow

Avoid this:

```
activation

↓

LoRA A

↓

LoRA B

↓

LoRA C

↓

return
```

Instead:

```
activation

↓

compute all deltas

↓

sum deltas

↓

activation += combined_delta

↓

return
```

One modification step.

Less overhead.

Cleaner scheduling.

---

# Layer Registration

Every TensorRT layer should receive a unique integer identifier.

Example:

```
0

Block0.q_proj

1

Block0.k_proj

2

Block0.v_proj

3

Block0.out_proj

4

Block0.fc1

5

Block0.fc2

...

N
```

The runtime should never search strings during inference.

Every adapter stores integer IDs.

Example:

```
layer_id = 143
```

Lookup becomes O(1).

---

# LoRA Storage

Each LoRA becomes:

```
Layer ID

↓

GPU buffers

↓

A matrix

↓

B matrix

↓

rank

↓

alpha

↓

strength
```

Example:

```python
layer_map = {

    31 : LoRALayer(...),

    47 : LoRALayer(...),

    103 : LoRALayer(...)
}
```

No string lookups during generation.

---

# GPU Execution

The runtime must never move activations to the CPU.

Correct flow:

```
TensorRT Layer

↓

GPU Activation

↓

CUDA LoRA Kernel

↓

GPU Activation

↓

TensorRT Layer
```

Never:

```
GPU

↓

CPU

↓

Python

↓

GPU
```

That would destroy TensorRT performance.

---

# Runtime Responsibilities

The Adapter Runtime should only:

- determine active adapters
- determine affected layers
- schedule execution
- launch CUDA kernels
- return updated activations

It should never execute TensorRT itself.

---

# Future Adapter Types

The following should require no runtime redesign:

- LoRA
- LyCORIS
- DoRA
- ControlNet
- IPAdapter
- T2I Adapter
- Reference Adapter

Each simply derives from RuntimeAdapter.

---

# ComfyUI Integration

Existing nodes should remain mostly unchanged.

Example:

```
Load TRT Engine

↓

Load LoRA

↓

Load Second LoRA

↓

Generate
```

Internally:

```
Load TRT Engine

↓

Engine Manager

↓

Adapter Manager

↓

Register LoRA

↓

Generate
```

No engine rebuild.

No TensorRT reload.

---

# Performance Targets

Target overhead:

No adapters

≈ 100% TensorRT speed

One LoRA

<5% slowdown

Multiple LoRAs

Scale efficiently

No engine rebuild

No model reload

GPU-only execution

Dynamic enable/disable

Hot swapping

Millisecond adapter registration

---

# Implementation Order

Phase 1

- RuntimeAdapter base class
- AdapterManager
- Layer ID registration

Phase 2

- LoRA parser
- GPU tensor storage
- Runtime registration

Phase 3

- CUDA LoRA execution kernel
- Activation modification

Phase 4

- Multi-LoRA scheduling
- Delta fusion
- Performance profiling

Phase 5

- Future adapter implementations
- ControlNet
- IPAdapter
- LyCORIS
- DoRA

---

# Final Design Principle

The TensorRT engine should remain completely unaware that adapters exist.

The Adapter Runtime should function as a GPU-resident middleware layer that intercepts activation flow, applies runtime modifications, and returns updated activations without requiring engine recompilation or altering the TensorRT execution pipeline.