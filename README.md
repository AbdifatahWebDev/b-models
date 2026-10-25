# b-models

Model files for **Brekwol Upscale**, an offline photo upscaler for Android. The app downloads
them from this repository's releases; nothing else is hosted here.

Each release holds ncnn model pairs (`.param` + `.bin`) and a `models.json` that lists every
file with its size and SHA-256. The app checks both before it installs a model.

| Model | Files | Use | License |
|---|---|---|---|
| Real-ESRGAN x4plus | `realesrgan-x4plus.*` | Everyday photos | BSD-3-Clause (Real-ESRGAN), MIT (ncnn conversion) |
| Real-ESRGAN Lite (x2plus) | `realesrgan-x2plus.*` | Direct 2× (built by `tools/lite/build.py`, workflow "Real-ESRGAN Lite") | BSD-3-Clause (Real-ESRGAN) |
| Real-ESRGAN Anime | `realesrgan-x4plus-anime.*` | Illustrations, comics, anime | BSD-3-Clause (Real-ESRGAN), MIT (ncnn conversion) |
| SwinIR | `swinir-x4.*` | Finest detail and edges | Apache-2.0 |
| SCUNet | `scunet-color.*` | Noisy, compressed or old scans | Apache-2.0 (code), MIT (KAIR weights release) |
| GFPGAN v1.4 (from `models-v2`) | `gfpgan-1.4.*` | Face restore | Apache-2.0 (code); weights trained on FFHQ (non-commercial dataset) |
| RetinaFace MobileNet 0.25 (from `models-v2`) | `retinaface-mnet.*` | Finds faces for GFPGAN and CodeFormer | MIT (facexlib) |
| CodeFormer (from `models-v3`) | `codeformer.*` | Face restore with a likeness control | **S-Lab License 1.0: non-commercial use only** (`LICENSE-CodeFormer.txt`) |
| AdcSR (from `models-v3`) | `adcsr-x4.*` | Generative upscale (slow, small photos) | Apache-2.0 (code); weights from SD 2.1-base, CreativeML Open RAIL++-M |

## Sources

- Real-ESRGAN by Xintao Wang et al. — https://github.com/xinntao/Real-ESRGAN.
  The ncnn files are unchanged from `realesrgan-ncnn-vulkan-20220424` in the Real-ESRGAN v0.2.5.0
  release (https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan, MIT).
- SwinIR by Jingyun Liang et al. — https://github.com/JingyunLiang/SwinIR.
  Converted to ncnn from the authors' `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN` weights,
  with fp16 weights and a fixed 128×128 input.
- SCUNet by Kai Zhang et al. — https://github.com/cszn/SCUNet.
  Converted to ncnn from the authors' `scunet_color_real_psnr` weights (published in
  https://github.com/cszn/KAIR, MIT), with fp16 weights and a
  fixed 256×256 input.
- GFPGAN by Xintao Wang et al. — https://github.com/TencentARC/GFPGAN (`GFPGANv1.4.pth`), and
  RetinaFace from facexlib — https://github.com/xinntao/facexlib (`detection_mobilenet0.25_Final.pth`).
  Converted by `tools/face/convert.py` in the **Face models** workflow, which checks the ncnn
  output against PyTorch. Run it from the Actions tab to rebuild `models-v2`.

- CodeFormer by Shangchen Zhou et al. — https://github.com/sczhou/CodeFormer (`codeformer.pth`,
  v0.1.0), converted by `tools/codeformer/convert.py` in the **CodeFormer** workflow, which
  checks the ncnn output against the original PyTorch model. The codebook lookup is rewritten
  as a sharpened softmax and the fidelity weight is a model input; the results are the same.
  Non-commercial use only: the app offers it in personal builds and leaves it out of store
  builds.

- AdcSR by Bin Chen et al. — https://github.com/Guaishou74851/AdcSR (`net_params_200.pkl`,
  `halfDecoder.ckpt`), converted by `tools/adcsr/convert.py` in the **AdcSR test** workflow;
  the **Models v3** workflow adds it to `models-v3`.

The converted SwinIR and SCUNet files are modified versions of the original works: the
window-attention layers were rewritten so ncnn can run them, with the same results (checked
against the original PyTorch models). The full license texts are attached to each release
as `LICENSE-*.txt`.
