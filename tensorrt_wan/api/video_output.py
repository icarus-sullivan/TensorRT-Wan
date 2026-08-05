"""VideoOutput: the return type of `WanEngine.generate()`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class VideoOutput:
    """`frames`: (T, H, W, C) uint8 in [0, 255], RGB."""

    frames: torch.Tensor
    fps: int

    def save(self, path: str | Path) -> None:
        """Write to an mp4 via `imageio` (ffmpeg backend). Not a hard dependency of this package
        since most workflows will save through ComfyUI's own video-saving nodes instead.
        """
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError(
                "VideoOutput.save() requires imageio with the ffmpeg backend: "
                "pip install 'imageio[ffmpeg]'"
            ) from exc

        iio.imwrite(str(path), self.frames.numpy(force=True), fps=self.fps, codec="libx264")

    def as_numpy(self):
        return self.frames.numpy(force=True)
