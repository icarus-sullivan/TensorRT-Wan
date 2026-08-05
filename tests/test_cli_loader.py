import pytest

from tensorrt_wan.cli.loader import resolve_loader


def test_resolve_loader_finds_stdlib_function():
    # os.getcwd takes no path argument, but this only checks resolution, not invocation.
    fn = resolve_loader("os:getcwd")
    assert fn is __import__("os").getcwd


def test_resolve_loader_rejects_missing_colon():
    with pytest.raises(ValueError, match="module.path:function_name"):
        resolve_loader("os.getcwd")


def test_resolve_loader_rejects_unknown_attribute():
    with pytest.raises(ValueError, match="no attribute"):
        resolve_loader("os:not_a_real_function")
