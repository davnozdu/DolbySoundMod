#!/usr/bin/env python3
"""Split StemgenRT vocals.onnx into a CPU/NPU pipeline for the Hexagon NPU (QNN HTP).

usage: make_npu_model.py vocals.onnx OUTDIR [--hops 15]

writes OUTDIR/vocals_pre.onnx, vocals_backbone.onnx, vocals_post.onnx

Why a split: run whole on HTP the model is fp16 end to end, and the 1024-point
FFT front end then buries the quiet top octave in rounding noise; the recurrent
state drifts and the output turns to garbage within seconds (measured on device).
So the exact and cheap parts stay on the CPU in fp32 and only the heavy part
(the recurrent backbone, ~75% of the CPU time) goes to the NPU:

  pre      (CPU)  audio + history          -> spectral features   (real FFT)
  backbone (NPU)  features + GRU states    -> mask                 (fp16 on HTP)
  post     (CPU)  mask + audio + history   -> separated vocals     (inverse FFT)

Features and post-processing depend only on audio, never on the network state,
so every stage is unrolled `hops` times: AudioFlinger hands the effect 1920
frames (15 hops) at once and the whole pipeline runs as three calls per block -
the NPU wakes 25 times a second instead of 375, which is what saves battery.

STATUS: research tool, not used by the build. Measured on the phone (SM8850,
Hexagon V81, QNN 2.42 via ORT 1.25.1 QNN EP): backbone 1.7 ms per 40 ms block on
the NPU vs 6.8 ms on the CPU, and the CPU-only split matches the original to
-127 dB - but on the NPU the recurrent state drifts (error vs fp32: -40 dB in the
first second, -6..-14 dB after three), so the module keeps the model on the CPU.
Each op alone is within 0.1-0.5% on HTP; the drift is the GRU accumulating it.

The backbone is rewritten for HTP: shapes are folded to constants and every
tensor above rank 4 is re-expressed without its unit dimensions (HTP rejects the
5-7D transposes of the band-split encoder). Each block tensor is laid out as
[hops, *per_hop_shape]. The pipeline is checked against the original model
before anything is written.
"""

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HOP = 128
FEATURES = "/vocals/Transpose_output_0"
MASK = "/vocals/backbone/Concat_19_output_0"

def fold(model):
    """Constant-fold shape arithmetic with ONNX Runtime's basic (pure ONNX) passes."""
    import onnxruntime as ort
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.onnx"), os.path.join(d, "out.onnx")
        onnx.save(model, src)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = dst
        ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
        folded = onnx.load(dst)
    # ORT's fusions can leave several nodes with one name; ONNX requires unique names.
    for i, n in enumerate(folded.graph.node):
        n.name = f"{n.name or n.op_type}#{i}"
    return folded


ELEMENTWISE = {"Add", "Sub", "Mul", "Div", "Pow", "Sigmoid", "Tanh", "Sqrt", "Log", "Exp",
               "Neg", "Abs", "Relu", "Clip", "Identity", "Cast"}


def reduce_rank(model, max_rank=4):
    """Rewrite every tensor of rank > max_rank without its unit dimensions.

    HTP rejects the 5-7D Transposes/MatMuls of the band-split encoder
    (Invalid_Transpose_Order). Dropping or inserting size-1 dimensions never moves
    data, so each operand is simply Reshaped into the lower-rank layout an op needs
    and the op's axes are remapped. Checked numerically against the original.
    """
    from onnx import shape_inference
    model = shape_inference.infer_shapes(model)
    g = model.graph
    shapes = {}
    for v in list(g.value_info) + list(g.input) + list(g.output):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for name, arr in inits.items():
        shapes[name] = list(arr.shape)
    consts = {}
    for n in g.node:
        if n.op_type == "Constant":
            arr = numpy_helper.to_array(n.attribute[0].t)
            consts[n.output[0]] = arr
            shapes[n.output[0]] = list(arr.shape)
    cur = {}      # tensor -> shape it has in the rewritten graph (if changed)
    new_nodes = []
    counter = [0]

    def uid(base):
        counter[0] += 1
        return f"rr/{base}#{counter[0]}"

    def const_i64(values):
        name = uid("shape")
        g.initializer.append(numpy_helper.from_array(np.array(values, np.int64), name))
        return name

    def shape_now(t):
        return cur.get(t, shapes[t])

    def as_shape(t, want):
        """t reshaped to `want` (same element order)."""
        if list(shape_now(t)) == list(want):
            return t
        if t in inits or t in consts:
            src = inits.get(t, consts.get(t))
            name = uid("init")
            g.initializer.append(numpy_helper.from_array(src.reshape(want).astype(src.dtype), name))
            return name
        out = uid("reshape")
        new_nodes.append(helper.make_node("Reshape", [t, const_i64(want)], [out]))
        return out

    def aligned(t, rank):
        s = shapes[t]
        return [1] * (rank - len(s)) + list(s)

    def squeeze(s):
        return [d for d in s if d != 1] or [1]

    for n in g.node:
        if n.op_type == "Constant":
            new_nodes.append(n)
            continue
        io = [t for t in list(n.input) + list(n.output) if t]
        high = any(len(shapes.get(t, [])) > max_rank for t in io)
        if not high:
            # Inputs that were reduced go back to their original shape.
            ins = [as_shape(t, shapes[t]) if t in cur else t for t in n.input]
            c = onnx.NodeProto(); c.CopyFrom(n); del c.input[:]; c.input.extend(ins)
            new_nodes.append(c)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        out = n.output[0]
        O = shapes[out]
        op = n.op_type
        if op in ("Reshape", "Unsqueeze", "Squeeze", "Flatten"):
            target = squeeze(O) if len(O) > max_rank else O
            new_nodes.append(helper.make_node("Reshape", [n.input[0], const_i64(target)], [out]))
            if len(O) > max_rank:
                cur[out] = target
        elif op in ELEMENTWISE:
            r = len(O)
            keep = [i for i in range(r) if O[i] != 1] or [r - 1]
            ins = []
            for idx, t in enumerate(n.input):
                if not t:
                    ins.append(t)
                elif op == "Clip" and idx > 0:
                    ins.append(t)
                else:
                    a = aligned(t, r)
                    ins.append(as_shape(t, [a[i] for i in keep]))
            c = onnx.NodeProto(); c.CopyFrom(n); del c.input[:]; c.input.extend(ins)
            new_nodes.append(c)
            cur[out] = [O[i] for i in keep]
        elif op == "Transpose":
            t = n.input[0]
            I = shapes[t]
            perm = attrs["perm"]
            ki = [i for i in range(len(I)) if I[i] != 1]
            new_perm = [ki.index(p) for p in perm if I[p] != 1]
            src = as_shape(t, [I[i] for i in ki])
            new_nodes.append(helper.make_node("Transpose", [src], [out], perm=new_perm))
            cur[out] = [I[ki[p]] for p in new_perm]
        elif op == "MatMul":
            r = len(O)
            keep = [i for i in range(r - 2) if O[i] != 1]
            a, b = aligned(n.input[0], r), aligned(n.input[1], r)
            A = as_shape(n.input[0], [a[i] for i in keep] + a[-2:])
            B = as_shape(n.input[1], [b[i] for i in keep] + b[-2:])
            new_nodes.append(helper.make_node("MatMul", [A, B], [out]))
            cur[out] = [O[i] for i in keep] + O[-2:]
        elif op == "Concat":
            r = len(O)
            ax = attrs["axis"] % r
            keep = [i for i in range(r) if O[i] != 1 or i == ax]
            ins = [as_shape(t, [aligned(t, r)[i] for i in keep]) for t in n.input]
            new_nodes.append(helper.make_node("Concat", ins, [out], axis=keep.index(ax)))
            cur[out] = [O[i] for i in keep]
        elif op == "Slice":
            data = n.input[0]
            D = shapes[data]
            r = len(D)
            axes_name = n.input[3] if len(n.input) > 3 and n.input[3] else None
            axes = list(inits.get(axes_name, consts.get(axes_name))) if axes_name else list(range(len(inits.get(n.input[1], consts.get(n.input[1])))))
            axes = [int(x) % r for x in axes]
            keep = [i for i in range(r) if O[i] != 1 or i in axes]
            src = as_shape(data, [D[i] for i in keep])
            ins = [src, n.input[1], n.input[2], const_i64([keep.index(x) for x in axes])]
            if len(n.input) > 4 and n.input[4]:
                ins.append(n.input[4])
            new_nodes.append(helper.make_node("Slice", ins, [out]))
            cur[out] = [O[i] for i in keep]
        elif op == "Gather":
            data, idx = n.input
            D = shapes[data]
            r = len(D)
            ax = attrs.get("axis", 0) % r
            assert len(shapes[idx]) == 0, "only scalar gathers are rewritten"
            keep = [i for i in range(r) if D[i] != 1 or i == ax]
            src = as_shape(data, [D[i] for i in keep])
            new_nodes.append(helper.make_node("Gather", [src, idx], [out], axis=keep.index(ax)))
            res = [D[i] for i in keep if i != ax]
            if len(O) > max_rank:
                cur[out] = res
            elif res != O:
                tmp = uid("gather")
                new_nodes[-1].output[0] = tmp
                cur[tmp] = res
                shapes[tmp] = res
                new_nodes.append(helper.make_node("Reshape", [tmp, const_i64(O)], [out]))
        else:
            raise SystemExit(f"reduce_rank: no rule for {op} {n.name} {[shapes.get(t) for t in io]}")
        for t, s in list(cur.items()):
            if len(s) > max_rank:
                raise SystemExit(f"reduce_rank: {t} still rank {len(s)}: {s}")
    del g.node[:]
    g.node.extend(new_nodes)
    del g.value_info[:]
    return model



def extract(model, inputs, outputs):
    from onnx.utils import Extractor
    return Extractor(onnx.shape_inference.infer_shapes(model)).extract_model(inputs, outputs)


def dims(model, name):
    g = onnx.shape_inference.infer_shapes(model).graph
    for v in list(g.input) + list(g.output) + list(g.value_info):
        if v.name == name:
            return [d.dim_value for d in v.type.tensor_type.shape.dim]
    raise KeyError(name)


def unroll(model, hops, block_in, states, block_out, rename=None):
    """Unroll a one-hop model: block tensors are [hops, *per_hop_shape].

    block_in:  per-hop inputs, fed as one [hops, ...] tensor each
    states:    (input, output) pairs carried from hop to hop
    block_out: per-hop outputs, returned as one [hops, ...] tensor each
    rename:    public names for the block tensors
    """
    rename = rename or {}
    g = model.graph
    init_names = {i.name for i in g.initializer}
    shape = {n: dims(model, n) for n in list(block_in) + list(block_out) + [s for p in states for s in p]}
    nodes, extra_init = [], []

    def public(n):
        return rename.get(n, n)

    for name in block_in:
        s = shape[name]
        extra_init.append(numpy_helper.from_array(np.array(s, np.int64), f"u/{name}/shape"))
        nodes.append(helper.make_node(
            "Split", [public(name), f"u/{name}/split"], [f"u/h{h}/{name}/raw" for h in range(hops)], axis=0))
        extra_init.append(numpy_helper.from_array(np.ones(hops, np.int64), f"u/{name}/split"))
        for h in range(hops):
            nodes.append(helper.make_node("Reshape", [f"u/h{h}/{name}/raw", f"u/{name}/shape"], [f"u/h{h}/{name}"]))

    cur = {i: i for i, _ in states}
    outs = {n: [] for n in block_out}
    for h in range(hops):
        last = h == hops - 1
        m = {name: f"u/h{h}/{name}" for name in block_in}
        for (si, so) in states:
            m[si] = cur[si]
            m[so] = ("next_" + si) if last else f"u/h{h}/{so}"
        for idx, n in enumerate(g.node):
            c = onnx.NodeProto()
            c.CopyFrom(n)
            c.name = f"u/h{h}/{n.name or n.op_type}#{idx}"
            for k, i in enumerate(c.input):
                if i and i not in init_names:
                    c.input[k] = m.get(i, f"u/h{h}/{i}")
            for k, o in enumerate(c.output):
                if o:
                    c.output[k] = m.get(o, f"u/h{h}/{o}")
            nodes.append(c)
        for (si, so) in states:
            cur[si] = m[so]
        for name in block_out:
            src = m.get(name, f"u/h{h}/{name}")
            nodes.append(helper.make_node("Reshape", [src, f"u/{name}/oshape"], [f"u/h{h}/{name}/out"]))
            outs[name].append(f"u/h{h}/{name}/out")
    for name in block_out:
        extra_init.append(numpy_helper.from_array(np.array([1] + shape[name], np.int64), f"u/{name}/oshape"))
        nodes.append(helper.make_node("Concat", outs[name], [public(name)], axis=0))

    def vi(name, s):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, s)

    graph = helper.make_graph(
        nodes, f"{g.name or 'stage'}_x{hops}",
        [vi(public(n), [hops] + shape[n]) for n in block_in] + [vi(si, shape[si]) for si, _ in states],
        [vi(public(n), [hops] + shape[n]) for n in block_out] + [vi("next_" + si, shape[si]) for si, _ in states],
        initializer=list(g.initializer) + extra_init,
    )
    out = helper.make_model(graph, opset_imports=model.opset_import)
    out.ir_version = model.ir_version
    onnx.checker.check_model(out)
    return out


def build(src, hops):
    folded = fold(onnx.load(src))
    ins = [i.name for i in folded.graph.input]
    audio, history, local, glob, tail = ins
    outs = [o.name for o in folded.graph.output]
    sep, next_hist, next_local, next_glob, next_tail = outs

    pre = extract(folded, [audio, history], [FEATURES, next_hist])
    backbone = reduce_rank(extract(folded, [FEATURES, local, glob], [MASK, next_local, next_glob]))
    post = extract(folded, [MASK, audio, history, tail], [sep, next_hist, next_tail])

    names = {audio: "audio", FEATURES: "features", MASK: "mask", sep: "separated"}
    pre_u = unroll(pre, hops, [audio], [(history, next_hist)], [FEATURES], names)
    bb_u = unroll(backbone, hops, [FEATURES], [(local, next_local), (glob, next_glob)], [MASK], names)
    post_u = unroll(post, hops, [MASK, audio], [(history, next_hist), (tail, next_tail)], [sep], names)
    return pre_u, bb_u, post_u


def session(model):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])


def zeros(sess):
    return {i.name: np.zeros(i.shape, np.float32) for i in sess.get_inputs()}


def verify(src, stages, hops, seconds=3.0):
    """Run the three unrolled stages against the original one-hop model."""
    import onnxruntime as ort
    rng = np.random.default_rng(1)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ref = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    pre, bb, post = (session(m) for m in stages)
    n_blocks = int(seconds * 48000 / (hops * HOP))
    t = np.arange(n_blocks * hops * HOP) / 48000
    mono = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
    audio = np.stack([mono, np.roll(mono, 37)])

    ri = ref.get_inputs()
    rstate = [np.zeros(i.shape, np.float32) for i in ri[1:]]
    st_pre, st_bb, st_post = zeros(pre), zeros(bb), zeros(post)
    want, got = [], []
    for b in range(n_blocks):
        chunk = audio[:, b * hops * HOP:(b + 1) * hops * HOP]
        for h in range(hops):
            o = ref.run(None, {ri[0].name: chunk[None, :, h * HOP:(h + 1) * HOP],
                               **{i.name: s for i, s in zip(ri[1:], rstate)}})
            want.append(o[0].reshape(2, HOP))
            rstate = o[1:]
        blk = np.ascontiguousarray(chunk.reshape(2, hops, HOP).transpose(1, 0, 2))[:, None]  # [hops,1,2,128]
        st_pre["audio"] = blk
        o = dict(zip([x.name for x in pre.get_outputs()], pre.run(None, st_pre)))
        st_pre = {k: (o["next_" + k] if "next_" + k in o else v) for k, v in st_pre.items()}
        st_bb["features"] = o["features"].reshape(st_bb["features"].shape)
        o2 = dict(zip([x.name for x in bb.get_outputs()], bb.run(None, st_bb)))
        st_bb = {k: (o2["next_" + k] if "next_" + k in o2 else v) for k, v in st_bb.items()}
        st_post["mask"] = o2["mask"].reshape(st_post["mask"].shape)
        st_post["audio"] = blk
        o3 = dict(zip([x.name for x in post.get_outputs()], post.run(None, st_post)))
        st_post = {k: (o3["next_" + k] if "next_" + k in o3 else v) for k, v in st_post.items()}
        got.extend(o3["separated"].reshape(hops, 2, HOP))
    want, got = np.concatenate(want, 1), np.concatenate(got, 1)
    err = np.sqrt(np.mean((want - got) ** 2)) / (np.sqrt(np.mean(want ** 2)) + 1e-12)
    return 20 * np.log10(err + 1e-20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--hops", type=int, default=15)
    args = ap.parse_args()
    stages = build(args.src, args.hops)
    ops = sorted({n.op_type for n in stages[1].graph.node})
    err_db = verify(args.src, stages, args.hops)
    print(f"backbone ops: {ops}")
    print(f"pipeline x{args.hops} vs original: {err_db:.1f} dB")
    if err_db > -80:
        sys.exit("conversion changed the model output")
    os.makedirs(args.outdir, exist_ok=True)
    for name, m in zip(("vocals_pre", "vocals_backbone", "vocals_post"), stages):
        onnx.save(m, os.path.join(args.outdir, name + ".onnx"))
        size = os.path.getsize(os.path.join(args.outdir, name + ".onnx")) / 1e6
        print(f"  {name}.onnx: {len(m.graph.node)} nodes, {size:.1f} MB")


if __name__ == "__main__":
    main()
