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

## Progress

- 2026-11-12: step 1 written (klein_export.py --stage text, klein_ncnn.Pieces.text); tiny random
  self-test passes (cosine 0.9999). Gotchas: Qwen's RMSNorm is plain weight (Gemma's is 1 + w);
  HF applies the final norm to the last hidden state, so the tiny model has 28 layers.
  Workflow edit-model.yml (text job only) run #1 started on c47ea92.
- Phone, Create (v0.9.4): 255 s (text 9, steps 59, decode 179 in 9 tiles). Mali GPU in fp32 gave
  wrong pixels (diff 1.02) and was only 1.4x faster on a 16x16 piece: dropped, remove the trial.

## Design for step 2 (transformer), from reading transformer_flux2.py

One token stream x = [text (Nt = text_tokens, fixed) | image tokens | reference tokens], width D
= 3072. The phone computes RoPE cos/sin [N, 128] from ids (4 axes × 32, theta 2000,
get_1d_rotary_pos_embed with repeat_interleave_real: pairs are interleaved) and feeds them in.
Text ids are (0,0,0,i); image (0,y,x,0); reference k has t = 10·(k+1) (check _prepare_*_ids).

- klein-time: t -> [temb | double img mod (6D) | double txt mod (6D) | single mod (3D)]
  (timestep × 1000, Timesteps(256, flip_sin_to_cos) -> MLP; Flux2Modulation = Linear(SiLU(temb)),
  chunks of D: shift, scale, gate per set; no guidance embedder in klein 4B).
- klein-d0..d4: double blocks; d0 also holds x_embedder and context_embedder. In and out are the
  joined stream; inside split at Nt. LayerNorm without affine, (1 + scale)·norm + shift, joint
  attention over [text | image] with per-stream q/k/v projections and RMSNorm on q and k per
  head, gates, FF = linear_in -> SiLU(first half) × second half -> linear_out.
- klein-s0..: single blocks in groups (20 in all): to_qkv_mlp_proj -> q, k, v and the MLP half,
  attention, cat(attn, SwiGLU(mlp)) -> to_out, gate. Last piece: drop text, norm_out
  (AdaLayerNormContinuous with temb), proj_out -> [N image tokens, 128]; keep the first Ni.
- Attention written by hand for ncnn: no unbind/stack. Rotation as q·cos + (q @ R)·sin with R a
  constant 128×128 matrix (pair (a, b) -> (-b, a)); explicit matmul + softmax.
- Sequence lengths: trace two sizes (dynamic) and check in ncnn at Nt + 1024 (512 px), + 2048 and
  + 3072 (one and two references). Attention matrices at 3584 tokens are 1.2 GB in fp32.
- Later: the KV cache for reference tokens (Flux2KVCache: extract on step 1, reuse after), and
  whether fewer text tokens than 512 keep the quality.
- VAE: AutoencoderKLFlux2 encoder and decoder, tiled like SANA's tail if it is all local;
  latents patchified 2×2 and normalised with vae.bn running stats.
- Text run #2 (8-bit, block 64): real tokens cosine 0.9997 (min 0.9991) against fp32; padding
  tokens mean 0.974, min 0.47, but bf16 against fp32 alone gives padding min 0.76: padding is
  small (norms 28-145 against 640-760 for real tokens) and unstable. Accepted for now; the
  end-to-end picture comparison (step 3) decides. Text job: ~15 min, under 9 GB.
- Step 2 written: klein-time, klein-dNN, klein-sNN; tiny self-test exact (cosine 1.00000) at an
  untraced length. Gotchas: ncnn.Mat(numpy) must be .clone()d (unaligned memory crashed in-place
  layers in light mode); pnnx drops a plain x[:, nt:] slice (use tensor_split, and tie unused
  halves in so they do not become outputs); pieces are deep-copied to fp32 from a bf16 model.
- Run #3 (fac4ad8): both jobs pass. Text 17 min, 9 × 322 MB + 390 MB embedding; real tokens
  0.9996. Transformer 36 min, peak 13.6 GB: five double pieces of 261-286 MB, five single pieces
  of 521-560 MB, klein-time 304 MB (was not quantized; now is). One call against the bf16 model
  (512 text + 54 image tokens): cosine 0.9976, max difference 0.24 of a 3.0 range.
  So far about 7.6 GB.
- Next (step 3): VAE encoder/decoder export (whole at 512 px first; measure memory), sampler in
  klein_ncnn.py (sigmas linspace(1, 1/n, n) shifted by mu = compute_empirical_mu(tokens, steps),
  Euler steps on velocity, latents × bn_std + bn_mean, 2×2 unpatchify, decode), a check job that
  compares a picture and edits against the pipeline from the same noise, then the release.
- Step 3 written: klein_export.py --stage vae (klein-enc, klein-dec, meta-vae.json with the
  batch-norm mean/std), klein_ncnn.Pieces.encode/generate (sigmas, Euler, patchify), and
  klein_check.py (reference in PyTorch, then the pieces on the same embeddings' prompts and
  noise; fox from words and the dress try-on). Tiny self-test: `klein_export.py --random
  --check --stage all` matches Flux2KleinPipeline exactly, create and edit (max difference 0).
  Workflow: image job also exports the VAE; check job after both (or alone with artifacts_run).
- After the check: C++ byte-level BPE tokenizer for Qwen (vocab + merges file), release
  `edit-model`, engine/edit.cpp, Create screen model choice and "Add photos"; then the KV cache.
