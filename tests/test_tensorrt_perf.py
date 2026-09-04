"""Pure-logic tests for comfyui-wanrt/nodes/tensorrt_perf.py -- no CUDA/SageAttention3/ComfyUI
required.

`comfy.*`/`folder_paths`/`sageattn3` are only ever imported lazily inside functions (never at
module level -- see tensorrt_perf.py's docstring), so real torch is used directly and third-party
bits are faked/mocked per test instead of needing to be actually installed.

Loaded by file path rather than `from comfyui_wanrt.nodes import tensorrt_perf`: the directory is
named `comfyui-wanrt` (a hyphen, invalid in a Python import statement) on purpose -- see
comfyui-wanrt/__init__.py's docstring.
"""

import contextlib
import importlib.util
import io
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


perf = _load_module("tensorrt_perf", "comfyui-wanrt/nodes/tensorrt_perf.py")


class TestWanVariantInference(unittest.TestCase):
    def test_t2v_i2v_ti2v_detected_from_filename(self):
        self.assertEqual(perf.infer_wan_variant("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"), "wan2.2_t2v_14B")
        self.assertEqual(perf.infer_wan_variant("wan2.2_i2v_low_noise_14B.safetensors"), "wan2.2_i2v_14B")
        self.assertEqual(perf.infer_wan_variant("wan2.2_ti2v_5B.safetensors"), "wan2.2_ti2v_5B")

    def test_unrecognized_filename_returns_none(self):
        self.assertIsNone(perf.infer_wan_variant("some_random_checkpoint.safetensors"))

    def test_every_inferable_variant_has_a_mag_ratios_table(self):
        for _hint, variant in perf._WAN_VARIANT_HINTS:
            self.assertIn(variant, perf.WAN2_2_MAG_RATIOS)


class TestMagRatiosTables(unittest.TestCase):
    def test_tables_are_calibrated_for_an_even_step_count(self):
        # magcache_wanmodel_forward indexes as mag_ratios[cur_step*2 + cond_or_uncond_index],
        # so every table must have an even length.
        for name, table in perf.WAN2_2_MAG_RATIOS.items():
            self.assertEqual(len(table) % 2, 0, name)

    def test_tables_start_at_unity(self):
        for table in perf.WAN2_2_MAG_RATIOS.values():
            self.assertEqual(table[0], 1.0)
            self.assertEqual(table[1], 1.0)


class TestIsWanModel(unittest.TestCase):
    def test_matches_comfy_wan_module(self):
        class FakeWanModel:
            pass

        FakeWanModel.__module__ = "comfy.ldm.wan.model"
        self.assertTrue(perf.is_wan_model(FakeWanModel()))

    def test_rejects_other_architectures(self):
        class FakeFluxModel:
            pass

        FakeFluxModel.__module__ = "comfy.ldm.flux.model"
        self.assertFalse(perf.is_wan_model(FakeFluxModel()))


class TestMagCachePresets(unittest.TestCase):
    def test_presets_have_required_keys(self):
        for mode in ("Fast", "Balanced", "Quality"):
            params = perf._magcache_params(mode, custom=None)
            for key in ("magcache_thresh", "magcache_K", "retention_ratio", "start_step", "end_step"):
                self.assertIn(key, params)

    def test_quality_skips_less_aggressively_than_fast(self):
        fast = perf._magcache_params("Fast", None)
        quality = perf._magcache_params("Quality", None)
        self.assertLess(quality["magcache_K"], fast["magcache_K"])
        self.assertLess(quality["magcache_thresh"], fast["magcache_thresh"])

    def test_custom_passes_through_verbatim(self):
        custom = {"magcache_thresh": 0.5, "magcache_K": 6, "retention_ratio": 0.9, "start_step": 1, "end_step": 5}
        self.assertEqual(perf._magcache_params("Custom", custom), custom)


class TestGpuCompatCheck(unittest.TestCase):
    def test_no_cuda_raises(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                perf.check_sage3_gpu_compat()

    def test_recognized_blackwell_capability_passes(self):
        # thu-ml/SageAttention's sageattention3_blackwell subproject has a dedicated -gencode
        # branch for exactly these three (major, minor) pairs -- verified against its setup.py.
        for capability in ((10, 0), (12, 0), (12, 1)):
            with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                 mock.patch.object(torch.cuda, "get_device_capability", return_value=capability):
                perf.check_sage3_gpu_compat()  # must not raise

    def test_non_blackwell_capability_raises(self):
        # This node has no fallback to SpargeAttention/SageAttention2 -- any non-Blackwell GPU
        # (Ampere/Ada/Hopper included) must raise, not silently degrade.
        for capability in ((7, 5), (8, 6), (9, 0)):
            with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                 mock.patch.object(torch.cuda, "get_device_capability", return_value=capability), \
                 mock.patch.object(torch.cuda, "get_device_name", return_value="Fake non-Blackwell GPU"):
                with self.assertRaises(RuntimeError):
                    perf.check_sage3_gpu_compat()

    def test_unrecognized_blackwell_capability_raises(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 5)), \
             mock.patch.object(torch.cuda, "get_device_name", return_value="Fake Future Blackwell GPU"):
            with self.assertRaises(RuntimeError):
                perf.check_sage3_gpu_compat()


class TestSageAttn3Override(unittest.TestCase):
    """Exercises the reshape math and fallback behavior on CPU with a fake sageattn3 -- no real
    CUDA kernel involved."""

    def _build_override(self, sage3_fn):
        fake_module = mock.Mock()
        fake_module.sageattn3_blackwell = sage3_fn
        with mock.patch.object(perf, "check_sage3_gpu_compat"), \
             mock.patch.object(perf, "ensure_sageattn3", return_value=fake_module):
            return perf.build_sageattn3_override()

    def test_reshapes_to_hnd_and_back(self):
        seen = {}

        def fake_sage3(q, k, v, is_causal=False):
            seen["shape"] = tuple(q.shape)
            seen["is_causal"] = is_causal
            return q  # identity, just to check the output reshape round-trips

        override = self._build_override(fake_sage3)

        b, seq, heads, dim_head = 1, 256, 8, 64
        q = torch.randn(b, seq, heads * dim_head)
        k = torch.randn(b, seq, heads * dim_head)
        v = torch.randn(b, seq, heads * dim_head)

        def fallback_func(*args, **kwargs):
            raise AssertionError("fallback should not be used for a supported shape")

        out = override(fallback_func, q, k, v, heads)
        self.assertEqual(seen["shape"], (b, heads, seq, dim_head))
        self.assertFalse(seen["is_causal"])
        self.assertEqual(out.shape, (b, seq, heads * dim_head))

    def test_falls_back_on_short_sequence(self):
        def fake_sage3(*args, **kwargs):
            raise AssertionError("FP4 kernel should not be called for a too-short sequence")

        override = self._build_override(fake_sage3)

        b, seq, heads, dim_head = 1, 32, 8, 64  # seq < 128
        q = torch.randn(b, seq, heads * dim_head)
        k = torch.randn(b, seq, heads * dim_head)
        v = torch.randn(b, seq, heads * dim_head)

        fallback_called = {"done": False}

        def fallback_func(q, k, v, heads, **kwargs):
            fallback_called["done"] = True
            return q

        override(fallback_func, q, k, v, heads)
        self.assertTrue(fallback_called["done"])

    def test_falls_back_when_mask_present(self):
        def fake_sage3(*args, **kwargs):
            raise AssertionError("FP4 kernel should not be called when a mask is given")

        override = self._build_override(fake_sage3)

        b, seq, heads, dim_head = 1, 256, 8, 64
        q = torch.randn(b, seq, heads * dim_head)
        k = torch.randn(b, seq, heads * dim_head)
        v = torch.randn(b, seq, heads * dim_head)
        mask = torch.zeros(b, 1, seq, seq)

        fallback_called = {"done": False}

        def fallback_func(q, k, v, heads, mask=None, **kwargs):
            fallback_called["done"] = True
            self.assertIsNotNone(mask)
            return q

        override(fallback_func, q, k, v, heads, mask=mask)
        self.assertTrue(fallback_called["done"])

    def test_apply_sageattn_disabled_returns_model_unchanged(self):
        fake_model = mock.Mock()
        result = perf.apply_sageattn(fake_model, "Disabled")
        self.assertIs(result, fake_model)
        fake_model.clone.assert_not_called()


class TestMagCacheStateReset(unittest.TestCase):
    """Exercises apply_magcache's unet_wrapper_function reset-on-new-generation logic directly,
    without a real ComfyUI ModelPatcher or Wan model."""

    def _install(self, filename="wan2.2_t2v_high_noise_14B.safetensors", mode="Balanced"):
        fake_diffusion_model = mock.Mock()
        fake_diffusion_model.__class__.__module__ = "comfy.ldm.wan.model"

        fake_model_options = {"transformer_options": {}}

        fake_model = mock.Mock()
        fake_model.get_model_object.return_value = fake_diffusion_model
        fake_model.model_options = fake_model_options
        fake_model.clone.return_value = fake_model
        captured = {}
        fake_model.set_model_unet_function_wrapper.side_effect = lambda fn: captured.__setitem__("wrapper", fn)

        new_model = perf.apply_magcache(fake_model, filename, mode)
        return new_model, captured["wrapper"]

    def test_state_resets_at_step_zero(self):
        model, wrapper = self._install()
        to = model.model_options["transformer_options"]
        state = to["magcache_state"]
        # simulate a mid-run mutation from a previous generation
        state[0]["accumulated_err"] = 0.9
        state[0]["residual_cache"] = torch.zeros(1)

        sigmas = torch.tensor([1.0, 0.5, 0.0])
        kwargs = {
            "input": torch.zeros(1),
            "timestep": torch.tensor([1.0]),
            "cond_or_uncond": [0],
            "c": {"transformer_options": dict(to)},
        }
        kwargs["c"]["transformer_options"]["sample_sigmas"] = sigmas

        def model_function(input_, timestep, **c):
            return input_

        wrapper(model_function, kwargs)

        self.assertEqual(state[0]["accumulated_err"], 0.0)
        self.assertIsNone(state[0]["residual_cache"])

    def _run_first_step(self, wrapper, to, num_steps):
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        kwargs = {
            "input": torch.zeros(1),
            "timestep": sigmas[0:1],
            "cond_or_uncond": [0],
            "c": {"transformer_options": dict(to)},
        }
        kwargs["c"]["transformer_options"]["sample_sigmas"] = sigmas

        def model_function(input_, timestep, **c):
            return input_

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wrapper(model_function, kwargs)
        return buf.getvalue()

    def test_warns_once_for_lightx2v_style_few_step_run(self):
        model, wrapper = self._install()
        to = model.model_options["transformer_options"]

        output = self._run_first_step(wrapper, to, num_steps=8)
        self.assertIn("WARNING", output)
        self.assertIn("8 steps", output)

        # must not repeat the warning on a later step of the same run
        output2 = self._run_first_step(wrapper, to, num_steps=8)
        self.assertNotIn("WARNING", output2)

    def test_no_warning_for_a_normal_step_count(self):
        model, wrapper = self._install()
        to = model.model_options["transformer_options"]

        output = self._run_first_step(wrapper, to, num_steps=50)
        self.assertNotIn("WARNING", output)

    def test_raises_for_non_wan_model(self):
        fake_diffusion_model = mock.Mock()
        fake_diffusion_model.__class__.__module__ = "comfy.ldm.flux.model"
        fake_model = mock.Mock()
        fake_model.get_model_object.return_value = fake_diffusion_model

        with self.assertRaises(RuntimeError):
            perf.apply_magcache(fake_model, "some_flux_checkpoint.safetensors", "Balanced")

    def test_raises_when_variant_unrecognized(self):
        fake_diffusion_model = mock.Mock()
        fake_diffusion_model.__class__.__module__ = "comfy.ldm.wan.model"
        fake_model = mock.Mock()
        fake_model.get_model_object.return_value = fake_diffusion_model

        with self.assertRaises(RuntimeError):
            perf.apply_magcache(fake_model, "mystery_checkpoint.safetensors", "Balanced")

    def test_disabled_mode_returns_model_unchanged(self):
        fake_model = mock.Mock()
        result = perf.apply_magcache(fake_model, "wan2.2_t2v_14B.safetensors", "Disabled")
        self.assertIs(result, fake_model)
        fake_model.clone.assert_not_called()


if __name__ == "__main__":
    unittest.main()
