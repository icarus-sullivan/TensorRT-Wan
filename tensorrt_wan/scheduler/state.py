"""GPU-resident scheduler state.

Every tensor here lives on the same CUDA device as the latents for the whole denoising loop.
`step_index` is the one piece of state kept as a plain Python int rather than a 0-d GPU tensor:
it only ever drives Python-side indexing/control-flow (`timesteps[step_index]`), so keeping it on
GPU would force a device-to-host sync on every step for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SchedulerState:
    """Precomputed, GPU-resident schedule for one generation run.

    Built once by `Scheduler.prepare()` and mutated in place by `Scheduler.step()` — reusing
    these buffers instead of reallocating per step is what "reuse GPU buffers" in the project
    spec means concretely here.
    """

    timesteps: torch.Tensor  # (num_steps,) on device
    sigmas: torch.Tensor  # (num_steps + 1,) on device, flow-matching noise levels
    step_index: int = 0

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.shape[0])

    @property
    def done(self) -> bool:
        return self.step_index >= self.num_steps

    @property
    def current_timestep(self) -> torch.Tensor:
        return self.timesteps[self.step_index]

    def advance(self) -> None:
        self.step_index += 1
