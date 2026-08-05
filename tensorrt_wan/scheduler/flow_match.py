"""Flow-matching Euler scheduler — Wan's native sampling algorithm.

This is real, runnable scheduling math (no model weights, export, or GPU build involved), unlike
the engine/export modules which are structural stubs pending the RunPod validation phase. Mirrors
the discretization used by SD3/Wan: linear sigmas in [0, 1] with an optional resolution-dependent
shift, Euler-integrated in velocity space.
"""

from __future__ import annotations

import torch

from tensorrt_wan.scheduler.base import Scheduler
from tensorrt_wan.scheduler.state import SchedulerState


class FlowMatchEulerScheduler(Scheduler):
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0) -> None:
        """`shift` > 1 biases more sampling steps toward high noise, which flow-matching models
        (Wan included) benefit from at higher resolutions — matches the `shift` knob exposed by
        Wan's own FlowMatchEulerDiscreteScheduler so engine output is comparable step-for-step.
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

    def prepare(self, num_inference_steps: int, device: torch.device) -> SchedulerState:
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        timesteps = sigmas[:-1] * self.num_train_timesteps
        return SchedulerState(timesteps=timesteps, sigmas=sigmas)

    def step(
        self,
        state: SchedulerState,
        model_output: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        sigma = state.sigmas[state.step_index]
        sigma_next = state.sigmas[state.step_index + 1]
        new_latents = latents + (sigma_next - sigma) * model_output
        state.advance()
        return new_latents
