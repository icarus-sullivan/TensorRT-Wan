"""Text-to-video via the standalone Python API.

Not run as part of this repository (see PLAN.md's development rule: no inference is executed
until engines are built on GPU hardware). Requires a model directory produced by
`trtwan build engine` for each of text_encoder/dit/vae_encoder/vae_decoder — see
docs/engine_generation.md.
"""

from tensorrt_wan import WanEngine


def main() -> None:
    engine = WanEngine.from_pretrained("/path/to/built/Wan2.1-T2V-14B", precision="auto")

    video = engine.generate(
        prompt="a fox running through a snowy forest, cinematic lighting",
        num_frames=81,
        resolution=(480, 832),
        num_inference_steps=30,
        guidance_scale=5.0,
        seed=42,
    )
    video.save("fox_in_snow.mp4")


if __name__ == "__main__":
    main()
