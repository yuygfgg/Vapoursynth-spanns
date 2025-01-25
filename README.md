# SPANNS VapourSynth Plugin

This VapourSynth plugin implements the The Survival Probability Adapted Nuclear Norm Shrinkage (SPANNS) Denoiser. 

Also check [spanns](https://github.com/Gabriella-Chaos/spanns)

## Build Instructions

[Opencv](https://github.com/opencv/opencv), [Eigen](https://github.com/PX4/eigen) and [GSL](https://www.gnu.org/software/gsl/) are required.

```bash
meson setup build
ninja -C build
ninja -C build install
```

## Usage

```python
spans = core.spanns.SPANNS(VideoNode clip[, VideoNode ref1, VideoNode ref2, float sigma=1.0, float tol=0.7, float gamma=0.5, int passes=2])
```

## Parameter Description

- clip (VideoNode): The original video node. Only 32bit float input is supported.
- ref1 (VideoNode): Reference clip, an approximation of the result. Default is a median filter.
- ref2 (VideoNode): Reference clip, a blurred one obtained from the original. It serves as an alternative way to control denoising strength. If provided, sigma will be ignored. Default is a box blur with radius equal to sigma.
- sigma (float): Denoising strength. Default is 1.
- tol (float): Noise tolerance in the range [0, 1]. Default is 0.7.
- gamma (float): Texture threshold in the range [0, 1]. Higher values preserve less texture. Default is 0.5.
- passes (int): Number of denoising steps. Default is 2.

