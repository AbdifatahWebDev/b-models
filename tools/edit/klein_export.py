#!/usr/bin/env python3
"""FLUX.2 klein 4B -> ncnn pieces for Brekwol Upscale's edit model (see PLAN.md).

    python klein_export.py --stage text --repo black-forest-labs/FLUX.2-klein-4B --out DIST \
        --quantizer ncnnllm2int [--bits 8 --block 64]
    python klein_export.py --random --stage text --out DIST --check      # tiny self-test

Stage text (this file so far): the Qwen3 text encoder as the pipeline uses it
(Flux2KleinPipeline._get_qwen3_prompt_embeds): the chat-templated prompt padded to TOKENS tokens,
the hidden states after layers 9, 18 and 27 joined per token. Layers 0-26 go out in pieces of
three (qwen-00 .. qwen-08), each (x, cos, sin, mask) -> x, so every third piece's output is one
of the three wanted states; weights block-quantized by ncnnllm2int, activations fp32. The
embedding rows are int8 with a scale per row (qwen-embed.bin, tools/create's format).
meta-text.json has what the phone needs; klein_ncnn.py runs the pieces as the phone does.
"""
import argparse
import copy
import json
import os
import sys
import tempfile

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'create'))
from sana_export import MASK_NEG, pnnx_export, quantize, rope, rotate_half, write_embed  # noqa: E402

LAYERS_OUT = (9, 18, 27)


def rms(x, w, eps):
    """Qwen3RMSNorm (plain weight; Gemma's, in sana_export.py, scales by 1 + weight)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class QwenLayers(nn.Module):
    """Qwen3 decoder layers [first, last) on n tokens.

    forward(x [1, n, H], cos [1, n, D], sin [1, n, D], mask [1, n, n]) -> x [1, n, H]
    """

    def __init__(self, model, first, last):
        super().__init__()
        cfg = model.config
        self.layers = nn.ModuleList(copy.deepcopy(l).float() for l in model.layers[first:last])
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5
        self.eps = cfg.rms_norm_eps

    def forward(self, x, cos, sin, mask):
        n = x.shape[1]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rep = self.heads // self.kv_heads
        for layer in self.layers:
            a = layer.self_attn
            h = rms(x, layer.input_layernorm.weight, self.eps)
            # Qwen3 normalises queries and keys per head before RoPE.
            q = rms(a.q_proj(h).view(1, n, self.heads, self.dim), a.q_norm.weight, self.eps).transpose(1, 2)
            k = rms(a.k_proj(h).view(1, n, self.kv_heads, self.dim), a.k_norm.weight, self.eps).transpose(1, 2)
            v = a.v_proj(h).view(1, n, self.kv_heads, self.dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            # Grouped-query attention without repeat_interleave (see tools/create/sana_export.py).
            q = q.reshape(self.kv_heads, rep * n, self.dim)
            s = torch.matmul(q, k.squeeze(0).transpose(1, 2)) * self.scale  # [kv, rep*n, n]
            s = s.reshape(self.kv_heads, rep, n, n) + mask
            p = torch.softmax(s, dim=-1).reshape(self.kv_heads, rep * n, n)
            o = torch.matmul(p, v.squeeze(0))
            o = o.reshape(self.heads, n, self.dim).transpose(0, 1).reshape(1, n, self.heads * self.dim)
            x = x + a.o_proj(o)
            h = rms(x, layer.post_attention_layernorm.weight, self.eps)
            m = layer.mlp
            x = x + m.down_proj(nn.functional.silu(m.gate_proj(h)) * m.up_proj(h))
        return x


def prompt_mask(n, real):
    """Additive mask [n, n] as HF builds it for a right-padded prompt of [real] tokens: token i sees
    the real tokens up to its own position (a padding token sees every real token)."""
    m = torch.full((n, n), MASK_NEG)
    for i in range(n):
        m[i, :min(i + 1, real)] = 0
    return m


def chat(tokenizer, prompt):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def reference(model, tokenizer, prompt, tokens):
    """The pipeline's prompt_embeds [tokens, 3 * H] and its input ids and count of real tokens."""
    t = tokenizer(chat(tokenizer, prompt), return_tensors='pt', padding='max_length', truncation=True, max_length=tokens)
    with torch.no_grad():
        out = model(input_ids=t.input_ids, attention_mask=t.attention_mask, output_hidden_states=True, use_cache=False)
    hs = torch.stack([out.hidden_states[k].float() for k in LAYERS_OUT], dim=1)  # [1, 3, n, H]
    return hs.permute(0, 2, 1, 3).reshape(tokens, -1).numpy(), t.input_ids[0].tolist(), int(t.attention_mask.sum())


def random_text():
    """A tiny random Qwen3 and a real byte-level BPE tokenizer with Qwen's chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(['a red fox sitting in fresh snow in a forest', 'user assistant think dress the woman in image'] * 4,
                           trainers.BpeTrainer(vocab_size=400, special_tokens=['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<think>', '</think>'],
                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token='<|endoftext|>', padding_side='right')
    tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                         "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and not enable_thinking %}"
                         "<think>\n\n</think>\n\n{% endif %}{% endif %}")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=28,  # one more than used: HF applies the final norm to the last state
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    model = Qwen3Model(cfg).eval()
    for p in model.parameters():  # livelier than the default init, still tame over 27 layers
        if p.dim() > 1:
            nn.init.normal_(p, std=0.08)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='black-forest-labs/FLUX.2-klein-4B')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', choices=['text'], default='text')
    ap.add_argument('--random', action='store_true', help='tiny random models: tests the export and the pieces')
    ap.add_argument('--tokens', type=int, default=512, help="prompt tokens, padded (the pipeline's max_sequence_length)")
    ap.add_argument('--layers-per-piece', type=int, default=3)
    ap.add_argument('--quantizer', help='path to ncnnllm2int')
    ap.add_argument('--bits', type=int, default=8)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--check', action='store_true', help='run the pieces against PyTorch')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix='klein-')
    torch.set_grad_enabled(False)
    assert all(k % a.layers_per_piece == 0 for k in LAYERS_OUT), 'the wanted layers must end pieces'

    if a.random:
        model, tok = random_text()
        a.tokens = min(a.tokens, 48)
    else:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.repo, subfolder='tokenizer')
        model = AutoModel.from_pretrained(a.repo, subfolder='text_encoder', torch_dtype=torch.bfloat16).eval()
    cfg = model.config
    print(f'text encoder: {type(model).__name__}, {len(model.layers)} layers, hidden {cfg.hidden_size}, '
          f'heads {cfg.num_attention_heads}/{cfg.num_key_value_heads} x {cfg.head_dim}, vocab {cfg.vocab_size}, '
          f'padding {tok.padding_side}, pad id {tok.pad_token_id}')
    assert tok.padding_side == 'right', 'the pieces assume right padding'
    template = chat(tok, 'PROMPT')
    head, tail = template.split('PROMPT')
    print(f'chat template: {head!r} + prompt + {tail!r}')

    # The check's reference first, while the whole model is still here.
    prompt = 'a red fox sitting in fresh snow in a forest, sharp focus on the eyes'
    want, ids, real = reference(model, tok, prompt, a.tokens) if a.check else (None, None, None)

    emb_scale, emb_q = write_embed(model.embed_tokens.weight, os.path.join(a.out, 'qwen-embed.bin'))
    n, H = a.tokens, cfg.hidden_size
    last = max(LAYERS_OUT)
    pieces = []
    # A second reference for the check: the same layers in fp32 in PyTorch, fed the int8 embedding
    # rows. Against it the pieces show only what quantizing the weights costs; the model itself
    # runs in bf16, whose rounding shows up on the padding tokens.
    x32, kept32 = None, []
    if a.check:
        x32 = torch.from_numpy(emb_q[ids].astype(np.float32) * emb_scale[ids, None]).unsqueeze(0)
        cos32, sin32 = rope(cfg, range(n))
        mask32 = prompt_mask(n, real).unsqueeze(0)
    for first in range(0, last, a.layers_per_piece):
        m = QwenLayers(model, first, first + a.layers_per_piece)
        if a.check:
            x32 = m(x32, cos32.unsqueeze(0), sin32.unsqueeze(0), mask32)
            if first + a.layers_per_piece in LAYERS_OUT:
                kept32.append(x32[0].numpy().copy())
        stem = os.path.join(a.out, f'qwen-{len(pieces):02d}')
        cos, sin = rope(cfg, range(n))
        # --random: fp32 weights, so the check tests the graph, not rounding (as in sana_export.py).
        pnnx_export(m, (torch.zeros(1, n, H), cos.unsqueeze(0), sin.unsqueeze(0), prompt_mask(n, n // 2).unsqueeze(0)),
                    None, stem, work, fp16=not a.random)
        if a.quantizer:
            quantize(stem, a.quantizer, a.bits, a.block)
        pieces.append(os.path.basename(stem))
        del m
        print(f'exported {pieces[-1]} (layers {first}-{first + a.layers_per_piece - 1})', flush=True)

    meta = {
        'version': 1, 'text_tokens': n, 'hidden': H, 'head_dim': cfg.head_dim,
        'rope_theta': float(getattr(cfg, 'rope_theta', None) or cfg.rope_parameters['rope_theta']),
        'mask_neg': MASK_NEG, 'text_pieces': pieces,
        'text_outputs': [k // a.layers_per_piece - 1 for k in LAYERS_OUT],  # pieces whose output is kept
        'chat_head': head, 'chat_tail': tail, 'pad_id': tok.pad_token_id,
    }
    with open(os.path.join(a.out, 'meta-text.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    if a.check:
        import gc
        import klein_ncnn
        del model  # 8 GB the check does not need
        gc.collect()
        got = klein_ncnn.Pieces(a.out).text(ids, real)
        want32 = np.concatenate(kept32, axis=1)

        def cosines(x, y):
            return [float(np.dot(x[i], y[i]) / (np.linalg.norm(x[i]) * np.linalg.norm(y[i]) + 1e-12)) for i in range(n)]

        def report(name, c):
            print(f'{name}: real tokens min {min(c[:real]):.5f} mean {np.mean(c[:real]):.5f}; '
                  f'padding min {min(c[real:] or [1]):.5f} mean {np.mean(c[real:] or [1]):.5f}')

        print(f'text check, {real} real tokens of {n}')
        quant = cosines(got, want32)
        report('pieces against fp32 PyTorch (cost of quantizing)', quant)
        report('pieces against the model as loaded', cosines(got, want))
        report('fp32 PyTorch against the model as loaded', cosines(want32, want))
        norms = np.linalg.norm(want32.reshape(n, len(LAYERS_OUT), -1), axis=2)
        print('norms per kept layer, real tokens mean', norms[:real].mean(0).round(1), 'padding mean', norms[real:].mean(0).round(1))
        if min(quant[:real]) < (0.999 if a.random else 0.98) or min(quant) < 0.9:
            raise SystemExit('the text pieces do not match PyTorch')


if __name__ == '__main__':
    main()
