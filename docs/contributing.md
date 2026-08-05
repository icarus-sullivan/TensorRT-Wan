# Contributing

Thanks for considering a contribution to TensorRT-Wan.

## Before opening a PR

- Run `ruff check .`, `black --check .`, `mypy tensorrt_wan`, and `pytest` (see
  [developer_guide.md](developer_guide.md)).
- If your change touches export/plugin/engine code, note in the PR description whether it's been
  validated against real hardware (this repository's own development happened without GPU access
  — see PLAN.md — so a lot of existing code is marked as such; new contributions with GPU access
  should validate and say so).
- Keep changes scoped: this project optimizes the shared Wan backbone, not individual workflows
  (see [architecture.md](architecture.md)). A PR that adds workflow-specific logic outside
  `conditioning/sources/` is very likely the wrong shape for this codebase — open an issue first
  if you're unsure.

## Where things go

- New conditioning method -> `tensorrt_wan/conditioning/sources/` + register in whatever composes
  `ConditioningManager` (`api/wan_engine.py`, or a ComfyUI node).
- New custom TensorRT op -> `tensorrt_wan/plugins/csrc/` (see [plugins.md](plugins.md)).
- New CLI command -> `tensorrt_wan/cli/commands/`, added to `ALL_COMMANDS` in
  `cli/commands/__init__.py`.
- New ComfyUI node -> `comfyui/nodes/` (see [developer_guide.md](developer_guide.md)).

## Commit / PR style

Small, focused commits with a clear "why" in the message. Reference the PLAN.md section your
change implements when relevant.

## Reporting issues

Include: `trtwan gpu-report` output, TensorRT-Wan version (`tensorrt_wan.__version__`), and
whether the issue reproduces without a GPU (structural bug) or only during an actual
export/build/inference run (numerical/GPU-behavior bug — see
[troubleshooting.md](troubleshooting.md) first).

## Code of conduct

Be respectful, assume good faith, keep discussion technical.
