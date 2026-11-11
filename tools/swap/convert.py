#!/usr/bin/env python3
"""Converts FaceFusion's face swap models to ncnn for Brekwol Upscale's Swap screen.

    python tools/swap/convert.py --arcface arcface_w600k_r50.onnx \
        --hyperswap hyperswap_1a_256.onnx --out dist/ --check

Writes, into --out:
    arcface-w600k.param / .bin   112x112 RGB in [-1, 1] (arcface_112_v2 crop) -> 512 identity
    hyperswap-1a.param / .bin    in0: identity (1x512, L2-normalised), in1: 256x256 RGB in [-1, 1]
                                 (arcface_128 crop) -> out0: 256x256 RGB in [-1, 1], out1: mask

Both come from FaceFusion's model assets (github.com/facefusion/facefusion-assets): ArcFace
w600k_r50 (InsightFace, non-commercial) and HyperSwap 1a 256 (FaceFusion, ResearchRAIL).
hyperswap ships with fp16 weights and Cast nodes at its inputs and outputs; it is made fully fp32
first, so pnnx sees constant weights (otherwise its convolutions become dynamic-weight layers in
fp32), then converted with fp16 weights. With --check, both ncnn models are run against the
original ONNX models in onnxruntime.

Needs onnx, onnxruntime, numpy, pnnx and (for --check) ncnn.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def to_fp32(src, dst):
    """fp16 weights and constants -> fp32; Cast-to-fp16 nodes become Cast-to-fp32."""
    m = onnx.load(src)
    g = m.graph
    for i, t in enumerate(g.initializer):
        if t.data_type == TensorProto.FLOAT16:
            g.initializer[i].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
    for n in g.node:
        for at in n.attribute:
            if n.op_type == 'Constant' and at.name == 'value' and at.t.data_type == TensorProto.FLOAT16:
                at.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(at.t).astype(np.float32), at.t.name))
            if n.op_type == 'Cast' and at.name == 'to' and at.i == TensorProto.FLOAT16:
                at.i = TensorProto.FLOAT
    del g.value_info[:]
    onnx.checker.check_model(m)
    onnx.save(m, dst)


def pnnx(onnx_path, inputshape, out_stem):
    """Runs pnnx on [onnx_path] (fp16 weights) and moves its .ncnn.param/.bin to out_stem.param/.bin."""
    work = tempfile.mkdtemp()
    local = os.path.join(work, 'model.onnx')
    shutil.copy(onnx_path, local)
    subprocess.run(['pnnx', local, f'inputshape={inputshape}', 'fp16=1'], cwd=work, check=True,
                   stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(work, 'model.ncnn.param'), out_stem + '.param')
    shutil.move(os.path.join(work, 'model.ncnn.bin'), out_stem + '.bin')
    shutil.rmtree(work)


def run_ncnn(stem, feeds, outputs):
    import ncnn
    net = ncnn.Net()
    net.opt.use_fp16_storage = True
    net.opt.use_fp16_packed = True
    net.opt.use_fp16_arithmetic = False
    net.load_param(stem + '.param')
    net.load_model(stem + '.bin')
    ex = net.create_extractor()
    for name, a in feeds.items():
        ex.input(name, ncnn.Mat(np.ascontiguousarray(a[0])))
    res = []
    for name in outputs:
        r, m = ex.extract(name)
        assert r == 0, f'{stem}: extract {name} failed'
        res.append(np.array(m))
    return res


def check(arcface, hyperswap, out):
    import onnxruntime as ort
    rng = np.random.RandomState(0)
    x = rng.uniform(-1, 1, (1, 3, 112, 112)).astype(np.float32)
    ref = ort.InferenceSession(arcface).run(None, {'input': x})[0].ravel()
    got = run_ncnn(os.path.join(out, 'arcface-w600k'), {'in0': x}, ['out0'])[0].ravel()
    cos = float(got @ ref / np.linalg.norm(got) / np.linalg.norm(ref))
    print(f'arcface: cosine with onnxruntime {cos:.6f}')
    assert cos > 0.9999, 'arcface: ncnn differs from the ONNX model'

    s = rng.randn(1, 512).astype(np.float32)
    s /= np.linalg.norm(s)
    t = rng.uniform(-1, 1, (1, 3, 256, 256)).astype(np.float32)
    ref = ort.InferenceSession(hyperswap).run(None, {'source': s, 'target': t})
    got = run_ncnn(os.path.join(out, 'hyperswap-1a'), {'in0': s, 'in1': t}, ['out0', 'out1'])
    for name, g, r in zip(['image', 'mask'], got, ref):
        g = g.reshape(r[0].shape)
        d = np.abs(g - r[0])
        print(f'hyperswap {name}: max difference {d.max():.4f}, mean {d.mean():.6f} (the original runs in fp16)')
        assert d.max() < 0.05 and d.mean() < 0.001, f'hyperswap {name}: ncnn differs from the ONNX model'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arcface', required=True)
    ap.add_argument('--hyperswap', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp()
    hs32 = os.path.join(work, 'hyperswap_fp32.onnx')
    to_fp32(a.hyperswap, hs32)
    pnnx(a.arcface, '[1,3,112,112]', os.path.join(a.out, 'arcface-w600k'))
    pnnx(hs32, '[1,512],[1,3,256,256]', os.path.join(a.out, 'hyperswap-1a'))
    if a.check:
        check(a.arcface, a.hyperswap, a.out)
    shutil.rmtree(work)
    for f in sorted(os.listdir(a.out)):
        print(f, os.path.getsize(os.path.join(a.out, f)))


if __name__ == '__main__':
    sys.exit(main())
