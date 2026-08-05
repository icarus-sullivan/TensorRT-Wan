# Developer Guide

## Project layout

See [architecture.md](architecture.md)'s module map. Read that first — this page is about
day-to-day mechanics, not the design.

## Environment

```bash
pip install -e ".[dev,tensorrt]"
```

## Coding standards

- Type hints throughout; `from __future__ import annotations` in every module (lets modern
  union syntax like `X | None` work even where a dependency requires Python 3.10's minimum).
- Dataclasses for anything that's primarily state (`GPUInfo`, `CacheKey`, `UnifiedConditioning`,
  every `config/schema.py` section).
- Composition over inheritance: `ConditioningSource`/`ModelExporter`/`Scheduler` are the only
  abstract base classes with multiple implementations; everything else composes concrete
  collaborators (e.g. `DiTEngine` holds a `TensorRTEngineWrapper`, it doesn't subclass it).
- One `ConditioningSource` per conditioning kind, registered into `ConditioningManager` — never a
  branch on kind inside `DiTEngine` or the manager. See
  [architecture.md](architecture.md#why-conditioning-is-a-registry-not-a-branch).
- Docstrings explain *why*, not *what* — a function's name and type hints should already say what
  it does; the docstring earns its place by explaining a non-obvious constraint or design
  decision (see almost any file in `tensorrt_wan/runtime/` for the intended style).

## Linting / formatting / type-checking

```bash
ruff check .
black --check .
mypy tensorrt_wan
```

Configured in `pyproject.toml` (`[tool.ruff]`, `[tool.black]`, `[tool.mypy]`).

## Tests

```bash
pytest
```

Tests in `tests/` are written to run without a GPU where the code under test doesn't need one
(`config`, `runtime.gpu`'s CPU-safe path, `runtime.precision`, `runtime.cache`,
`runtime.fallback`, `conditioning`, `scheduler`, `cli.loader`) — see each test module's imports
for what it does/doesn't require. Nothing in `tests/` has been executed in this repository (see
PLAN.md's development rule); running them is the first thing to do once `torch` is installed in
your environment.

## Adding support for a new Wan release

1. If tensor dimensions changed, update the relevant `ModelExporter` subclass's constructor
   defaults/callers (see [export.md](export.md)) — dimensions are already parameters, not
   constants, so this is usually a caller-side change only.
2. If a new conditioning method shipped, add a `ConditioningSource` subclass under
   `conditioning/sources/` and register it (see [architecture.md](architecture.md)).
3. If a new fused op needs a custom kernel, add a plugin under `plugins/csrc/` (see
   [plugins.md](plugins.md)'s "Adding a new plugin" section).
4. Update `docs/supported_gpus.md`/`docs/optimization_strategy.md` if precision defaults change.

## Adding a new ComfyUI node

Add a module under `comfyui/nodes/` following any existing node as a template (standard
`INPUT_TYPES`/`RETURN_TYPES`/`FUNCTION`/`CATEGORY` ComfyUI node contract), export
`NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS` from it, and add it to `_MODULES` in
`comfyui/nodes/__init__.py`. Use **relative imports only** inside `comfyui/` — see
[comfyui_integration.md](comfyui_integration.md) for why.
