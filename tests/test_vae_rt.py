"""Pure-logic tests for comfyui-wanrt/nodes/vae_rt.py -- no TensorRT/CUDA/ComfyUI required.

`tensorrt`/`folder_paths` are only ever imported lazily inside functions (never at module level,
by design -- see vae_rt.py's docstring), so they're injected into sys.modules as fakes here rather
than needing to be actually installed.

Loaded by file path rather than `from comfyui_wanrt.nodes import vae_rt`: the directory is named
`comfyui-wanrt` (a hyphen, invalid in a Python import statement) on purpose, since that's the
literal folder name ComfyUI loads as a custom node package -- see comfyui-wanrt/__init__.py's
docstring for why every import inside that package is relative instead.
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


vae_rt = _load_module("vae_rt", "comfyui-wanrt/nodes/vae_rt.py")


def _fake_tensorrt_module(version: str = "10.5.0") -> types.ModuleType:
    fake = types.ModuleType("tensorrt")
    fake.__version__ = version
    return fake


def _fake_folder_paths_module(vae_files: list[str], vae_dir: str) -> types.ModuleType:
    fake = types.ModuleType("folder_paths")
    fake.models_dir = vae_dir
    fake.get_filename_list = lambda kind: list(vae_files) if kind == "vae" else []

    def _get_full_path_or_raise(kind, name):
        if kind == "vae" and name in vae_files:
            return str(Path(vae_dir) / name)
        raise FileNotFoundError(name)

    fake.get_full_path_or_raise = _get_full_path_or_raise
    fake.get_folder_paths = lambda kind: [vae_dir] if kind == "vae" else []
    return fake


class TestEnvelopeBounds(unittest.TestCase):
    def test_encoder_and_decoder_bounds_are_ordered(self):
        for bounds in (vae_rt.ENCODER_HEIGHT, vae_rt.ENCODER_WIDTH, vae_rt.DECODER_LATENT_HEIGHT, vae_rt.DECODER_LATENT_WIDTH):
            lo, opt, hi = bounds
            self.assertLess(lo, opt)
            self.assertLessEqual(opt, hi)

    def test_default_checkpoint_is_a_known_source(self):
        self.assertIn(vae_rt.DEFAULT_VAE_FILENAME, vae_rt.VAE_SOURCES)


_PROFILE_A = ((1, 3, 1, 256, 256), (1, 3, 1, 832, 832), (1, 3, 1, 1536, 1536))
_PROFILE_B = ((1, 3, 1, 256, 256), (1, 3, 1, 480, 832), (1, 3, 1, 1088, 1088))


class TestEngineFilename(unittest.TestCase):
    def test_deterministic_and_varies_with_inputs(self):
        with mock.patch.dict(sys.modules, {"tensorrt": _fake_tensorrt_module()}):
            a = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", 1, _PROFILE_A)
            b = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", 1, _PROFILE_A)
            c = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", 21, _PROFILE_A)
            d = vae_rt._engine_filename("vae_decoder", "wan_2.1_vae.safetensors", "fp16", 1, _PROFILE_A)

        self.assertEqual(a, b)
        self.assertNotEqual(a, c)  # different frame count -> different cached engine
        self.assertNotEqual(a, d)  # different component -> different cached engine
        self.assertTrue(a.endswith(".engine"))

    def test_filename_changes_with_trt_version(self):
        with mock.patch.dict(sys.modules, {"tensorrt": _fake_tensorrt_module("10.5.0")}):
            a = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", 1, _PROFILE_A)
        with mock.patch.dict(sys.modules, {"tensorrt": _fake_tensorrt_module("10.6.0")}):
            b = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", 1, _PROFILE_A)
        self.assertNotEqual(a, b)

    def test_filename_changes_with_profile_shape(self):
        """Regression test: a stale cached engine built under old resolution bounds must not be
        silently reused after ENCODER_HEIGHT/WIDTH or DECODER_LATENT_HEIGHT/WIDTH change."""
        with mock.patch.dict(sys.modules, {"tensorrt": _fake_tensorrt_module()}):
            a = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", 1, _PROFILE_A)
            b = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", 1, _PROFILE_B)
        self.assertNotEqual(a, b)


class TestAvailableVaeNames(unittest.TestCase):
    def test_includes_downloadable_sources_even_when_not_yet_downloaded(self):
        fake_fp = _fake_folder_paths_module(vae_files=[], vae_dir="/tmp/does-not-matter")
        with mock.patch.dict(sys.modules, {"folder_paths": fake_fp}):
            names = vae_rt._available_vae_names()
        self.assertIn(vae_rt.DEFAULT_VAE_FILENAME, names)
        self.assertIn("wan2.2_vae.safetensors", names)

    def test_default_checkpoint_sorts_first(self):
        fake_fp = _fake_folder_paths_module(vae_files=["a_custom_vae.safetensors"], vae_dir="/tmp")
        with mock.patch.dict(sys.modules, {"folder_paths": fake_fp}):
            names = vae_rt._available_vae_names()
        self.assertEqual(names[0], vae_rt.DEFAULT_VAE_FILENAME)


class TestEnsureVaeCheckpoint(unittest.TestCase):
    def test_returns_existing_path_without_downloading(self):
        fake_fp = _fake_folder_paths_module(vae_files=[vae_rt.DEFAULT_VAE_FILENAME], vae_dir="/tmp/vae")
        with mock.patch.dict(sys.modules, {"folder_paths": fake_fp}):
            path = vae_rt._ensure_vae_checkpoint(vae_rt.DEFAULT_VAE_FILENAME)
        self.assertEqual(path, Path("/tmp/vae") / vae_rt.DEFAULT_VAE_FILENAME)

    def test_unknown_filename_raises_without_network_access(self):
        fake_fp = _fake_folder_paths_module(vae_files=[], vae_dir="/tmp/vae")
        with mock.patch.dict(sys.modules, {"folder_paths": fake_fp}):
            with self.assertRaises(FileNotFoundError):
                vae_rt._ensure_vae_checkpoint("not_a_real_checkpoint.safetensors")


class TestChunkedDecode(unittest.TestCase):
    """Pure bookkeeping check for `_decode_chunked_trt`: does its overlap/trim arithmetic tile the
    chunk outputs to exactly the one-shot pixel-frame count, with no gap or double-count, using the
    real calibrated (a, b) = (4, -3) that matches this project's own known fact (81 pixel frames
    <-> 21 latent frames)? `_decode_chunk_trt` (the actual TensorRT call) is stubbed out -- this
    only exercises the chunk-splitting/trim math, not real decode correctness, which needs a GPU.
    """

    def _runtime_with_stubbed_chunks(self, calls: list[int]):
        runtime = vae_rt._VAERuntime.__new__(vae_rt._VAERuntime)  # skip __init__: no checkpoint/GPU needed
        runtime._temporal_upsample = (4, -3)

        def fake_decode_chunk(latent: torch.Tensor) -> torch.Tensor:
            frames = latent.shape[2]
            calls.append(frames)
            pixel_frames = 4 * frames - 3
            return torch.zeros(1, 3, pixel_frames, 4, 4)

        runtime._decode_chunk_trt = fake_decode_chunk
        return runtime

    def test_chunk_sizes_match_hand_derived_sequence_for_21_frames(self):
        calls: list[int] = []
        runtime = self._runtime_with_stubbed_chunks(calls)
        latent = torch.zeros(1, 1, 21, 4, 4)

        out = runtime._decode_chunked_trt(latent)

        self.assertEqual(calls, [9, 9, 9, 6])  # hand-derived: [0:9],[5:14],[10:19],[15:21]
        self.assertEqual(out.shape[2], 4 * 21 - 3)  # exactly matches the one-shot formula, no gap/overlap

    def test_short_sequence_within_one_chunk_makes_a_single_call(self):
        calls: list[int] = []
        runtime = self._runtime_with_stubbed_chunks(calls)
        latent = torch.zeros(1, 1, 7, 4, 4)

        out = runtime._decode_chunked_trt(latent)

        self.assertEqual(calls, [7])
        self.assertEqual(out.shape[2], 4 * 7 - 3)

    def test_exact_multiple_of_stride_has_no_trailing_short_chunk(self):
        # frames=9+5+5=19 lands exactly on a chunk boundary (stride = CHUNK-OVERLAP = 5).
        calls: list[int] = []
        runtime = self._runtime_with_stubbed_chunks(calls)
        latent = torch.zeros(1, 1, 19, 4, 4)

        out = runtime._decode_chunked_trt(latent)

        self.assertEqual(calls, [9, 9, 9])
        self.assertEqual(out.shape[2], 4 * 19 - 3)


if __name__ == "__main__":
    unittest.main()
