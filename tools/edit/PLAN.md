# Edit model: FLUX.2 klein 4B for Brekwol Upscale (plan and facts)

Goal: a second Create model that also edits photos with reference images (dress from a PNG,
background, restyle), like the klein column in the create-test release. Apache-2.0, not gated.

## Facts (checked 2026-11-12)

- Pipeline: `diffusers.Flux2KleinPipeline` (diffusers 0.42.1, `pipelines/flux2/pipeline_flux2_klein.py`).
  Distilled: 4 steps, no guidance (guidance_scale ignored, `guidance=None`), FlowMatch Euler with
  `mu = compute_empirical_mu(image_seq_len, steps)`.
- Text encoder: Qwen3-4B (36 layers, hidden 2560, MLP 9728 SiLU, 32 heads / 8 KV heads, head 128,
  vocab 151936, RoPE theta 1e6, RMSNorm 1e-6). Chat template with `enable_thinking=False`,
  max 512 tokens, hidden states of layers 9, 18, 27 stacked -> 7680 per token. Only the first 27
  layers are needed.
- Transformer: `Flux2Transformer2DModel`, 5 double + 20 single blocks, 24 heads × 128 = 3072,
  in_channels 128, joint_attention_dim 7680, mlp_ratio 3, RoPE axes [32,32,32,32] theta 2000,
  no guidance embedding. Reference images are extra tokens with t-coordinate 10, 20, ...
- VAE: `AutoencoderKLFlux2`, 8× with 2×2 patchify (128 latent channels after patchify), latents
  normalised with the VAE's batch-norm running mean/var. Encoder needed for reference images.
- Test server (4 vCPU, PyTorch fp32, 512 px): text-to-image ~420 s, one-reference edit ~640 s,
  two-reference try-on ~880 s, text encoding ~50 s a prompt, peak 15.7 GB.
- Phone (Redmi Note 13 Pro+, 2 big cores): SANA-Sprint 0.6B takes 344-445 s where the server's
  PyTorch took ~90 s: expect klein at 512 px around 20-30 min a picture, about an hour a try-on.
- `Flux2KleinKVPipeline` (pipeline_flux2_klein_kv.py) caches the reference tokens' keys and values
  after the first step: worth using for edits (reference tokens are most of the sequence).

## What the phone already taught us (Create, SANA)

- fp16 storage overflows (values in the millions) -> everything fp32 on ARM, or int8 block
  quantization of weights with fp32 activations (ncnnllm2int), which worked for Gemma (cos 0.9996).
- Trace at small sizes with two shapes (dynamic), check in ncnn at the real size; avoid stack,
  repeat_interleave, movedim, 5-D+ tensors, `k[0]` indexing (see tools/create/sana_export.py).
- ncnn.Mat(numpy) shares memory: keep arrays alive.
- All 8 cores are not faster than the 2 big ones; the Mali GPU trial result is still open.
- Load one piece at a time; a 16 GB CI runner needs one job per model and small traces.

## Steps

1. `klein_export.py --stage text`: Qwen3 layers 0-26 in pieces of two or three layers, int8 blocks,
   fixed token count (prompt padded), outputs after layers 9, 18, 27; vocabulary + merges for a
   C++ byte-level BPE tokenizer; embedding rows int8. Check against PyTorch (cosine).
2. `--stage image`: transformer in pieces (double blocks, then single blocks), int8 blocks; inputs
   image tokens + reference tokens + text tokens; VAE encoder and decoder (tiled like SANA's tail).
3. `klein_ncnn.py`: the phone's pipeline in Python against PyTorch; samples for create and edits.
4. Workflow `edit-model.yml` (text / image / check jobs), release `edit-model`.
5. App: engine/edit.cpp (+ tokenizer), Create screen gets a model choice and "Add photos".
