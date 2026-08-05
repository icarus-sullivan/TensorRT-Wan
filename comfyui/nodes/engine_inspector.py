from pathlib import Path


class TensorRTEngineInspector:
    """Reports basic metadata (size, cached invalidation key, I/O tensor list if the `tensorrt`
    package is importable) for a built `.engine` file — the ComfyUI-graph form of
    `trtwan inspect`.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"engine_path": ("STRING", {"default": ""})}}

    def run(self, engine_path: str):
        path = Path(engine_path)
        if not path.exists():
            return (f"No engine file at {path}",)

        lines = [f"Engine: {path}", f"Size: {path.stat().st_size / (1 << 20):.1f} MiB"]

        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            lines.append(f"Cache metadata: {meta_path.read_text()}")

        try:
            import tensorrt as trt

            trt_logger = trt.Logger(trt.Logger.WARNING)
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(path.read_bytes())
            lines.append(f"I/O tensors: {engine.num_io_tensors}")
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                lines.append(f"  {name}: {engine.get_tensor_dtype(name)} {engine.get_tensor_shape(name)}")
        except ImportError:
            lines.append("(install the 'tensorrt' package for layer-level inspection)")

        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {"TensorRTEngineInspector": TensorRTEngineInspector}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTEngineInspector": "TensorRT Engine Inspector"}
