import torch

from tensorrt_wan.export.base import DynamicAxis, ModelExporter


class _FakeExporter(ModelExporter):
    """Minimal concrete ModelExporter for exercising shape_digest() without a real Wan module."""

    name = "fake"

    def example_inputs(self) -> dict[str, torch.Tensor]:
        return {"x": torch.zeros(1, 3, 480, 832)}

    def dynamic_axes(self) -> dict[str, list[DynamicAxis]]:
        if self.static:
            return {}
        return {"x": [DynamicAxis(name="dim2", min=64, opt=480, max=1920)]}

    @property
    def input_names(self) -> list[str]:
        return ["x"]

    @property
    def output_names(self) -> list[str]:
        return ["y"]


def _model() -> torch.nn.Module:
    return torch.nn.Linear(1, 1)


def test_shape_digest_differs_between_static_and_dynamic_at_identical_example_shape():
    # Regression test: static=True changes dynamic_axes() (empty dict) without changing
    # example_inputs()'s shapes at all -- a shapes-only digest used to hash identically for a
    # static and a wide-dynamic-range build of the same nominal shape, so `build engine` silently
    # served the stale dynamic-range engine instead of building the requested static one. Real bug
    # hit on RunPod hardware (vae_encoder/vae_decoder static rebuild request served the old
    # dynamic-range engine). See docs/wan2.2_i2v_14b_notes.md's 2026-08-06 session.
    dynamic = _FakeExporter(_model(), static=False)
    static = _FakeExporter(_model(), static=True)
    assert dynamic.example_inputs()["x"].shape == static.example_inputs()["x"].shape
    assert dynamic.shape_digest() != static.shape_digest()


def test_shape_digest_stable_for_equal_exporters():
    a = _FakeExporter(_model(), static=False)
    b = _FakeExporter(_model(), static=False)
    assert a.shape_digest() == b.shape_digest()


if __name__ == "__main__":
    test_shape_digest_differs_between_static_and_dynamic_at_identical_example_shape()
    test_shape_digest_stable_for_equal_exporters()
    print("ok")
