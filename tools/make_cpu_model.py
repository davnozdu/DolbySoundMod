#!/usr/bin/env python3
"""Split StemgenRT vocals.onnx into the neural backbone and its DSP parameters.

usage: make_cpu_model.py vocals.onnx OUTDIR

writes OUTDIR/vocals_backbone.onnx  features + GRU states -> mask   (ONNX Runtime)
       OUTDIR/vocals_dsp.bin        analysis/synthesis windows, constants (dsp/vocals.cpp)

ONNX Runtime spends ~37% of the model's time in the front and back end - three
1024-point DFTs per 128-sample hop through its generic DFT kernel. The module
does that part itself with one real FFT per hop (the analysis spectrum is shared
by the features and the mask), so the model file only keeps the backbone.

Front end (per channel):  X = rfft(window * [history | chunk])
                          s = sqrt(max(mean(|X|^2), CLIP_MIN))     (over both channels)
                          features = [Re X / s, Im X / s, log(1 + sqrt(|X|^2 + EPS) / s)]
Back end:                 y = irfft(X * mask)[768:1024] * synthesis_window
                          out = (y[:128] + tail) / denominator,  tail = y[128:]

Every graph pattern this relies on is asserted, and the C++ re-implementation is
checked against the full model in CI (dsp/test_vocals_dsp.cpp).

vocals_dsp.bin layout (little-endian float32 unless noted):
  char[8] "ALLYVDSP", uint32 version=1,
  float analysis[1024], float synthesis[256], float denominator[128],
  float eps_add, float clip_min
"""

import os
import struct
import sys
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper

FEATURES = "/vocals/Transpose_output_0"
MASK = "/vocals/backbone/Concat_19_output_0"


def fold(model):
    import onnxruntime as ort
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.onnx"), os.path.join(d, "out.onnx")
        onnx.save(model, src)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = dst
        ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
        out = onnx.load(dst)
    for i, n in enumerate(out.graph.node):
        n.name = f"{n.name or n.op_type}#{i}"
    return out


def extract(model, inputs, outputs):
    from onnx.utils import Extractor
    return Extractor(onnx.shape_inference.infer_shapes(model)).extract_model(inputs, outputs)


def constants(model):
    g = model.graph
    c = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    c.update({n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in g.node if n.op_type == "Constant"})
    return g, c


def operand(g, c, op, shape=None, dtype=np.float32, scalar=False):
    """The single constant operand of `op` nodes matching the shape/scalar filter."""
    found = []
    for n in g.node:
        if n.op_type != op:
            continue
        for i in n.input:
            if i in c and c[i].dtype == dtype:
                v = c[i]
                if scalar and v.size == 1 or shape is not None and tuple(v.shape) == tuple(shape):
                    found.append(v)
    uniq = {v.tobytes(): v for v in found}
    assert len(uniq) == 1, f"{op} {shape}: expected one constant, found {len(uniq)}"
    return next(iter(uniq.values()))


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    folded = fold(onnx.load(src))
    ins = [i.name for i in folded.graph.input]
    outs = [o.name for o in folded.graph.output]
    audio, history, local, glob, tail = ins
    sep, next_hist, next_local, next_glob, next_tail = outs

    pre = extract(folded, [audio, history], [FEATURES, next_hist])
    post = extract(folded, [MASK, audio, history, tail], [sep, next_hist, next_tail])
    backbone = extract(folded, [FEATURES, local, glob], [MASK, next_local, next_glob])

    gp, cp = constants(pre)
    gq, cq = constants(post)
    ops_pre = [n.op_type for n in gp.node if n.op_type != "Constant"]
    ops_post = [n.op_type for n in gq.node if n.op_type != "Constant"]
    assert ops_pre.count("DFT") == 1 and ops_post.count("DFT") == 2, (ops_pre, ops_post)
    assert ops_pre.count("Log") == 1 and ops_pre.count("ReduceMean") == 1, ops_pre

    analysis = operand(gp, cp, "Mul", shape=(1024,))
    assert np.array_equal(analysis, operand(gq, cq, "Mul", shape=(1024,))), "pre/post windows differ"
    adds = sorted({float(c.ravel()[0]) for n in gp.node if n.op_type == "Add"
                   for i in n.input if i in cp and cp[i].size == 1 for c in [cp[i]]})
    assert adds == sorted({adds[0], 1.0}) and len(adds) == 2 and adds[0] < 1e-6, adds  # eps and the log1p 1.0
    eps_add = adds[0]
    clip_min = float(operand(gp, cp, "Clip", scalar=True).ravel()[0])
    synthesis = operand(gq, cq, "Mul", shape=(256,))
    denominator = operand(gq, cq, "Div", shape=(128,))
    # Spectral weights only zero the imaginary part of DC and Nyquist, which an
    # inverse real FFT ignores anyway; the mirror gather builds the Hermitian half.
    weights = operand(gq, cq, "Mul", shape=(513, 2))
    expect = np.ones((513, 2), np.float32)
    expect[0, 1] = expect[512, 1] = 0.0
    assert np.array_equal(weights, expect), "unexpected spectral weights"
    mirror = operand(gq, cq, "Gather", shape=(511,), dtype=np.int64)
    assert np.array_equal(mirror, np.arange(511, 0, -1)), "unexpected mirror"

    os.makedirs(outdir, exist_ok=True)
    onnx.save(backbone, os.path.join(outdir, "vocals_backbone.onnx"))
    with open(os.path.join(outdir, "vocals_dsp.bin"), "wb") as f:
        f.write(b"ALLYVDSP" + struct.pack("<I", 1))
        for arr in (analysis, synthesis, denominator):
            f.write(np.ascontiguousarray(arr, "<f4").tobytes())
        f.write(struct.pack("<ff", eps_add, clip_min))
    print(f"backbone: {len(backbone.graph.node)} nodes, inputs {[i.name for i in backbone.graph.input]}")
    print(f"dsp: eps_add={eps_add:g} clip_min={clip_min:g}")


if __name__ == "__main__":
    main()
