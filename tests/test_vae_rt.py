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


class TestEngineFilename(unittest.TestCase):
    def test_deterministic_and_varies_with_inputs(self):
        with mock.patch.dict(sys.modules, {"tensorrt": _fake_tensorrt_module()}):
            a = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", frames=1)
            b = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", frames=1)
            c = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", frames=21)
            d = vae_rt._engine_filename("vae_decoder", "wan_2.1_vae.safetensors", "fp16", frames=1)

        self.assertEqual(a, b)
        self.assertNotEqual(a, c)  # different frame count -> different cached engine
        self.assertNotEqual(a, d)  # different component -> different cached engine
        self.assertTrue(a.endswith(".engine"))

    def test_filename_changes_with_trt_version(self):
        with mock.patch.dict(sys.modules, {"tensorrt": _fake_tensorrt_module("10.5.0")}):
            a = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", frames=1)
        with mock.patch.dict(sys.modules, {"tensorrt": _fake_tensorrt_module("10.6.0")}):
            b = vae_rt._engine_filename("vae_encoder", "wan_2.1_vae.safetensors", "fp16", frames=1)
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


if __name__ == "__main__":
    unittest.main()
