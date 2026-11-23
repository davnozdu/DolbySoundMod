#!/usr/bin/env python3
"""Prepare Essentia's Discogs-EffNet export for the Hexagon HTP.

usage: prepare_effnet.py discogs-effnet-bsdynamic-1.onnx discogs-effnet-htp.onnx

The QNN execution provider rejects the export's Identity node, which only feeds the
second output ("embeddings"); auto EQ reads "activations". This drops that output
and its Identity, and fixes the dynamic batch to the three patches dsp/auto_eq.cpp
always sends, so the graph is static. The weights and every remaining node are
untouched; the result is checked against the original on random input.
"""

import sys

import numpy as np
import onnx

PATCHES = 3


def main():
    src, dst = sys.argv[1], sys.argv[2]
    model = onnx.load(src)
    graph = model.graph
    dropped = [o for o in graph.output if o.name == "embeddings"]
    assert dropped and [o.name for o in graph.output] == ["activations", "embeddings"], \
        [o.name for o in graph.output]
    graph.output.remove(dropped[0])
    identities = [n for n in graph.node if n.op_type == "Identity" and "embeddings" in n.output]
    assert len(identities) == 1, identities
    graph.node.remove(identities[0])
    assert not any(n.op_type == "Identity" for n in graph.node), "unexpected Identity left"
    for value in list(graph.input) + list(graph.output):
        dim = value.type.tensor_type.shape.dim[0]
        assert dim.dim_param == "batch_size", (value.name, dim)
        dim.Clear()
        dim.dim_value = PATCHES
    onnx.checker.check_model(model)
    onnx.save(model, dst)

    import onnxruntime as ort
    x = np.random.default_rng(1).standard_normal((PATCHES, 128, 96)).astype(np.float32)
    want = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(["activations"], {"melspectrogram": x})[0]
    got = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(["activations"], {"melspectrogram": x})[0]
    err = float(np.abs(want - got).max())
    print(f"{dst}: {len(graph.node)} nodes, outputs {[o.name for o in graph.output]}, max difference {err:.2e}")
    if err > 1e-6:
        sys.exit("prepared Discogs-EffNet differs from the original")


if __name__ == "__main__":
    main()
