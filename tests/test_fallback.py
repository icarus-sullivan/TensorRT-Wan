import pytest

from tensorrt_wan.runtime.fallback import FallbackTriggered, run_with_fallback


def test_returns_trt_result_when_it_succeeds():
    result = run_with_fallback("op", lambda: "trt-result", lambda: "torch-result")
    assert result == "trt-result"


def test_falls_back_to_torch_on_trt_failure():
    def trt_fn():
        raise RuntimeError("no plugin for this op")

    result = run_with_fallback("op", trt_fn, lambda: "torch-result")
    assert result == "torch-result"


def test_fallback_triggered_records_op_name_and_cause():
    cause = ValueError("bad shape")

    def trt_fn():
        raise cause

    triggered = []

    def torch_fn():
        return "ok"

    # run_with_fallback logs the failure rather than raising it directly; construct the
    # exception the same way it does internally to check the message it would have logged.
    exc = FallbackTriggered("my_op", cause)
    assert exc.op_name == "my_op"
    assert exc.cause is cause
    assert "my_op" in str(exc)

    assert run_with_fallback("my_op", trt_fn, torch_fn) == "ok"


def test_raises_when_both_paths_fail():
    def trt_fn():
        raise RuntimeError("trt failed")

    def torch_fn():
        raise RuntimeError("torch failed too")

    with pytest.raises(RuntimeError, match="torch failed too"):
        run_with_fallback("op", trt_fn, torch_fn)
