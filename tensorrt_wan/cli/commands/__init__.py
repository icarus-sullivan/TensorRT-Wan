from tensorrt_wan.cli.commands import build, cache, export, gpu_report, inspect, list_engines, optimization_report

ALL_COMMANDS = (gpu_report, cache, export, build, inspect, list_engines, optimization_report)

__all__ = ["ALL_COMMANDS"]
