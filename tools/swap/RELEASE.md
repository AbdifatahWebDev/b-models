Face swap for Brekwol Upscale 0.8 (the Swap screen), converted to ncnn by `tools/swap/convert.py`
and checked against the original ONNX models in the workflow run:

- `hyperswap-1a.*` — HyperSwap 1a 256 (FaceFusion, 2025), fp16 weights. Redraws a 256 px aligned
  face crop with a new identity; the app runs it four times per face (pixel boost 512), as
  FaceFusion does.
- `arcface-w600k.*` — ArcFace w600k_r50 (InsightFace), fp16 weights. Reads the identity of the face
  that is put on.

The faces themselves are found by RetinaFace (`retinaface-mnet.*` in models-v3), and swapped faces
are finished with GFPGAN v1.4 (also models-v3).

Licences: HyperSwap is released by FaceFusion under its ResearchRAIL licence, and the ArcFace
weights by InsightFace for non-commercial research use. Both stay under those terms here. Use them
only with photos of people who agreed to it.
