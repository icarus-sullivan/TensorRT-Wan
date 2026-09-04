"""Pure-logic tests for comfyui-wanrt/nodes/rife_rt.py -- no TensorRT/CUDA/ComfyUI required.

Loaded by file path (the directory is named `comfyui-wanrt`, a hyphen -- invalid in a Python
import statement, on purpose; see comfyui-wanrt/__init__.py's docstring).
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rife_rt = _load_module("rife_rt", "comfyui-wanrt/nodes/rife_rt.py")


def _fake_tensorrt_module(version: str = "10.5.0") -> types.ModuleType:
    fake = types.ModuleType("tensorrt")
    fake.__version__ = version
    return fake


class _StubRuntime:
    """Fakes `_RifeRuntime.infer_pair` so `_interpolate_batch`'s loop/cache-clear logic can be
    tested without a real TensorRT engine. Returns a frame tagged with (pair_index, timestep) so
    the test can assert exact call order and count."""

    def __init__(self):
        self.calls: list[tuple[float, float]] = []
        self.device = torch.device("cpu")

    def infer_pair(self, frame_0: torch.Tensor, frame_1: torch.Tensor, timestep: float) -> torch.Tensor:
        self.calls.append((float(frame_0[0, 0, 0, 0]), timestep))
        return torch.full_like(frame_0, fill_value=-1.0)


class TestInterpolateBatch(unittest.TestCase):
    def test_multiplier_2_inserts_one_frame_per_gap(self):
        # 3 input frames, each a distinct constant value so output order is verifiable.
        frames = torch.stack([torch.full((3, 4, 4), v) for v in (0.0, 1.0, 2.0)])
        runtime = _StubRuntime()

        out = rife_rt._interpolate_batch(runtime, frames, multiplier=2, clear_cache_after_n_frames=100)

        # 3 originals + 2 interpolated (one per gap) = 5 frames, originals preserved untouched.
        self.assertEqual(out.shape[0], 5)
        self.assertTrue(torch.equal(out[0], frames[0]))
        self.assertTrue(torch.equal(out[2], frames[1]))
        self.assertTrue(torch.equal(out[4], frames[2]))
        self.assertEqual(len(runtime.calls), 2)
        self.assertAlmostEqual(runtime.calls[0][1], 0.5)

    def test_multiplier_4_inserts_three_frames_per_gap(self):
        frames = torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)])
        runtime = _StubRuntime()

        out = rife_rt._interpolate_batch(runtime, frames, multiplier=4, clear_cache_after_n_frames=100)

        self.assertEqual(out.shape[0], 5)  # 2 originals + 3 interpolated
        timesteps = [t for _, t in runtime.calls]
        self.assertEqual(timesteps, [0.25, 0.5, 0.75])

    def test_single_frame_gap_never_calls_the_model(self):
        frames = torch.zeros(1, 3, 4, 4)
        runtime = _StubRuntime()

        out = rife_rt._interpolate_batch(runtime, frames, multiplier=2, clear_cache_after_n_frames=100)

        self.assertEqual(out.shape[0], 1)
        self.assertEqual(runtime.calls, [])


class TestPadToMultiple(unittest.TestCase):
    def test_already_aligned_returns_input_unchanged(self):
        frame = torch.zeros(1, 3, 128, 192)
        padded, h, w = rife_rt._pad_to_multiple(frame, multiple=64)
        self.assertIs(padded, frame)
        self.assertEqual((h, w), (128, 192))

    def test_pads_up_to_next_multiple_on_right_and_bottom_only(self):
        frame = torch.rand(1, 3, 100, 130)
        padded, h, w = rife_rt._pad_to_multiple(frame, multiple=64)
        self.assertEqual((h, w), (100, 130))
        self.assertEqual(padded.shape[-2:], (128, 192))
        # Original content stays anchored at the top-left -- padding only extends right/bottom.
        self.assertTrue(torch.equal(padded[:, :, :h, :w], frame))
        self.assertTrue(torch.equal(padded[:, :, h:, :], torch.zeros(1, 3, 28, 192)))

    def test_crop_after_pad_round_trips_to_original_shape(self):
        frame = torch.rand(1, 3, 90, 150)
        padded, h, w = rife_rt._pad_to_multiple(frame, multiple=64)
        cropped = padded[:, :, :h, :w]
        self.assertTrue(torch.equal(cropped, frame))


class TestResampleFps(unittest.TestCase):
    def test_16_to_24_fps_inserts_two_interpolated_frames_for_a_3_frame_clip(self):
        frames = torch.stack([torch.full((3, 4, 4), v) for v in (0.0, 1.0, 2.0)])
        runtime = _StubRuntime()

        out = rife_rt._resample_fps(runtime, frames, source_fps=16.0, target_fps=24.0, clear_cache_after_n_frames=100)

        self.assertEqual(out.shape[0], 4)
        self.assertTrue(torch.equal(out[0], frames[0]))
        self.assertTrue(torch.equal(out[-1], frames[-1]))
        self.assertEqual(len(runtime.calls), 2)
        timesteps = sorted(t for _, t in runtime.calls)
        self.assertAlmostEqual(timesteps[0], 1 / 3, places=3)
        self.assertAlmostEqual(timesteps[1], 2 / 3, places=3)

    def test_equal_fps_is_a_passthrough_with_no_model_calls(self):
        frames = torch.stack([torch.full((3, 4, 4), v) for v in (0.0, 1.0, 2.0)])
        runtime = _StubRuntime()

        out = rife_rt._resample_fps(runtime, frames, source_fps=16.0, target_fps=16.0, clear_cache_after_n_frames=100)

        self.assertEqual(out.shape[0], 3)
        self.assertEqual(runtime.calls, [])
        for i in range(3):
            self.assertTrue(torch.equal(out[i], frames[i]))

    def test_upsampling_never_extrapolates_past_the_last_frame(self):
        frames = torch.stack([torch.full((3, 4, 4), v) for v in (0.0, 1.0)])
        runtime = _StubRuntime()

        out = rife_rt._resample_fps(runtime, frames, source_fps=10.0, target_fps=37.0, clear_cache_after_n_frames=100)

        self.assertTrue(torch.equal(out[0], frames[0]))
        self.assertTrue(torch.equal(out[-1], frames[-1]))


class TestEnvelopeBounds(unittest.TestCase):
    def test_bounds_are_ordered(self):
        self.assertLess(rife_rt.IMAGE_DIM_MIN, rife_rt.IMAGE_DIM_OPT)
        self.assertLessEqual(rife_rt.IMAGE_DIM_OPT, rife_rt.IMAGE_DIM_MAX)

    def test_default_model_is_a_known_option(self):
        self.assertIn(rife_rt.DEFAULT_RIFE_MODEL, rife_rt.RIFE_MODELS)


class TestEngineFilename(unittest.TestCase):
    def test_deterministic_and_varies_with_precision(self):
        with mock.patch.dict(sys.modules, {"tensorrt": _fake_tensorrt_module()}):
            a = rife_rt._engine_filename(rife_rt.DEFAULT_RIFE_MODEL, "fp16")
            b = rife_rt._engine_filename(rife_rt.DEFAULT_RIFE_MODEL, "fp16")
            c = rife_rt._engine_filename(rife_rt.DEFAULT_RIFE_MODEL, "fp32")

        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(a.endswith(".engine"))


if __name__ == "__main__":
    unittest.main()
