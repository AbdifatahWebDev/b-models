# b-models

Model files for **Brekwol Upscale**, an offline photo upscaler for Android. The app downloads
them from this repository's releases; nothing else is hosted here.

Each release holds ncnn model pairs (`.param` + `.bin`) and a `models.json` that lists every
file with its size and SHA-256. The app checks both before it installs a model.

| Model | Files | Use | License |
|---|---|---|---|
| Real-ESRGAN x4plus | `realesrgan-x4plus.*` | Everyday photos | BSD-3-Clause (Real-ESRGAN), MIT (ncnn conversion) |
| Real-ESRGAN Anime | `realesrgan-x4plus-anime.*` | Illustrations, comics, anime | BSD-3-Clause (Real-ESRGAN), MIT (ncnn conversion) |
| SwinIR | `swinir-x4.*` | Finest detail and edges | Apache-2.0 |
| SCUNet | `scunet-color.*` | Noisy, compressed or old scans | Apache-2.0 (code), MIT (KAIR weights release) |

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

The converted SwinIR and SCUNet files are modified versions of the original works: the
window-attention layers were rewritten so ncnn can run them, with the same results (checked
against the original PyTorch models). The full license texts are attached to each release
as `LICENSE-*.txt`.
