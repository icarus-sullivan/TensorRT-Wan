"""Image-to-video: the same `WanEngine.generate()` call as text-to-video, plus `image=` and/or
`last_image=` frame tensors. See docs/python_api.md — I2V/T2V share one code path by design
(PLAN.md's "one unified engine, not one per workflow" principle).
"""

import torch
import torchvision

from tensorrt_wan import WanEngine


def load_frame(path: str, device: torch.device) -> torch.Tensor:
    """Read an image file into (1, 3, H, W) float in [-1, 1], the VAE encoder's expected input."""
    image = torchvision.io.read_image(path).float() / 255.0  # (3, H, W) in [0, 1]
    image = image.unsqueeze(0).to(device) * 2.0 - 1.0
    return image


def main() -> None:
    engine = WanEngine.from_pretrained("/path/to/built/Wan2.1-I2V-14B", precision="auto")
    first_frame = load_frame("first_frame.png", engine.device)
    last_frame = load_frame("last_frame.png", engine.device)  # optional — omit for first-frame-only I2V

    video = engine.generate(
        prompt="the fox turns and looks at the camera",
        image=first_frame,
        last_image=last_frame,
        num_frames=81,
        resolution=(480, 832),
        num_inference_steps=30,
        guidance_scale=5.0,
        seed=42,
    )
    video.save("fox_turns.mp4")


if __name__ == "__main__":
    main()
