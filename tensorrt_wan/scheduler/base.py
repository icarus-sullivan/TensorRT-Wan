"""Scheduler interface.

Kept separate from the DiT engine so swapping the sampling algorithm (flow-match Euler today,
UniPC or a future solver later) never touches engine/conditioning code — the engine only ever
sees "give me the next timestep, take the model output, give me updated latents."
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from tensorrt_wan.scheduler.state import SchedulerState


class Scheduler(ABC):
    @abstractmethod
    def prepare(self, num_inference_steps: int, device: torch.device) -> SchedulerState:
        """Precompute a GPU-resident `SchedulerState` for a run of `num_inference_steps`."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        state: SchedulerState,
        model_output: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one denoising update, advance `state` in place, and return the new latents.

        Implementations must not introduce a device-to-host sync (no `.item()`/`.cpu()` on
        per-step tensors) — that is the entire reason this exists instead of using a stock
        `diffusers` scheduler directly, which CPU-syncs on `step_index` bookkeeping.
        """
        raise NotImplementedError
