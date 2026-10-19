Everything in `models-v1`, plus face restoration for Brekwol Upscale 0.3:

- `gfpgan-1.4.*` — GFPGAN v1.4 (TencentARC, Apache-2.0), converted to ncnn with fp16 weights
  and a fixed 512×512 input by `tools/face/convert.py`.
- `retinaface-mnet.*` — RetinaFace with a MobileNet 0.25 backbone (facexlib, MIT), fixed
  640×640 input. It finds the faces GFPGAN restores.

Both conversions were checked against the original PyTorch models in the workflow run.
GFPGAN's weights were trained on FFHQ, which is licensed for non-commercial use only.
