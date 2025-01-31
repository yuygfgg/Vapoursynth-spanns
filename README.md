# SPANNS VapourSynth Plugin

This VapourSynth plugin implements the The Survival Probability Adapted Nuclear Norm Shrinkage (SPANNS) Denoiser. 

Also check [spanns](https://github.com/Gabriella-Chaos/spanns)

## Build Instructions

[Opencv](https://github.com/opencv/opencv), [Eigen](https://github.com/PX4/eigen), [Boost](https://www.boost.org/) and [GSL](https://www.gnu.org/software/gsl/) are required.

```bash
meson setup build
ninja -C build
ninja -C build install
```

## Usage

```
core.spanns.SPANNS(VideoNode clip[, VideoNode ref, float sigma=1.0, float tol=0.0, float cutoff=0.9])
```

## Parameter Description

- clip (VideoNode): The original video node. Only 32bit float input is supported.
- ref (VideoNode): Reference clip, a blurred one obtained from the original. It serves as an alternative way to control denoising strength. If provided, sigma will be ignored. Default is a 3-pass box blur with radius equal to sigma.
- sigma (float): Denoising strength. Default is 1.
- tol (float): Noise tolerance in the range [0, 1]. Default is 0.0.
- cutoff (float): Information components cutoff point in the range $[0.5, 1) \cup [1, \infty)$. Values falls in the first interval represents a percentile cutoff, whereas the second interval represents the absolute index. Higher values preserve less texture. Default is 0.9.

