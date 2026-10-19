Everything in `models-v2`, plus the generative upscaler for Brekwol Upscale 0.4:

- `adcsr-x4.*` — AdcSR (CVPR 2025, Apache-2.0 code), a one-step diffusion 4× upscaler distilled
  from OSEDiff, converted to ncnn with fp16 weights and a fixed 128×128 input by
  `tools/adcsr/convert.py` (checked against PyTorch: 1.4 % max error). Its weights derive from
  Stable Diffusion 2.1-base and carry the CreativeML Open RAIL++-M license (commercial use
  allowed, with use-based restrictions).
