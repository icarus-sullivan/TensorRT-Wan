import torch

from tensorrt_wan.scheduler.flow_match import FlowMatchEulerScheduler


def test_prepare_produces_expected_step_count():
    scheduler = FlowMatchEulerScheduler()
    state = scheduler.prepare(10, torch.device("cpu"))
    assert state.num_steps == 10
    assert state.sigmas.shape[0] == 11
    assert not state.done


def test_sigmas_are_monotonically_decreasing():
    scheduler = FlowMatchEulerScheduler(shift=3.0)
    state = scheduler.prepare(20, torch.device("cpu"))
    assert torch.all(state.sigmas[:-1] >= state.sigmas[1:])
    assert torch.isclose(state.sigmas[-1], torch.tensor(0.0), atol=1e-6)


def test_step_advances_state_and_updates_latents():
    scheduler = FlowMatchEulerScheduler()
    state = scheduler.prepare(4, torch.device("cpu"))
    latents = torch.zeros(1, 4)
    model_output = torch.ones(1, 4)

    new_latents = scheduler.step(state, model_output, latents)

    assert state.step_index == 1
    assert not torch.equal(new_latents, latents)


def test_scheduler_completes_after_num_steps():
    scheduler = FlowMatchEulerScheduler()
    state = scheduler.prepare(3, torch.device("cpu"))
    latents = torch.zeros(1, 2)
    for _ in range(3):
        latents = scheduler.step(state, torch.zeros(1, 2), latents)
    assert state.done
