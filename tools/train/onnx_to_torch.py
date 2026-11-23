"""Rebuild the released StemgenRT vocal model (PyTorch) from its ONNX export.

usage: onnx_to_torch.py STEMGEN_SRC vocals.onnx OUT.pt [--wav song.wav]
The anonymous MatMul initializers are the PyTorch weights with the last two axes
swapped; named initializers map 1:1. The result is checked against ONNX Runtime.
"""
import sys, wave
import numpy as np, onnx, onnxruntime as ort, torch
from onnx import numpy_helper

src, onnx_path, out_path = sys.argv[1:4]
wav = sys.argv[sys.argv.index('--wav') + 1] if '--wav' in sys.argv else None
sys.path.insert(0, src)
from stemgenrt.banded import BandSeparator

model = BandSeparator(sources=("vocals",), band_width=96, global_width=192, layers=2)
sd = model.state_dict()
g = onnx.load(onnx_path).graph
ini = {i.name: numpy_helper.to_array(i) for i in g.initializer}
P = 'children_by_source.vocals.model.'
consumer = {}
for n in g.node:
    for i in n.input:
        consumer.setdefault(i, n.name)
new = {}
for k, v in sd.items():
    if P + k in ini:
        new[k] = torch.from_numpy(ini[P + k].copy())
for name, arr in ini.items():
    if not name.startswith('onnx::'):
        continue
    node = consumer[name]                     # /vocals/backbone/<module>/MatMul
    key = node.split('/vocals/')[1].rsplit('/', 1)[0].replace('/', '.') + '.weight'
    new[key] = torch.from_numpy(np.ascontiguousarray(np.swapaxes(arr, -1, -2)))
buffers = {k for k, _ in model.named_buffers()}
missing = [k for k in sd if k not in new and k not in buffers]
assert not missing, missing
for k in sd:
    if k in new:
        assert tuple(new[k].shape) == tuple(sd[k].shape), (k, new[k].shape, sd[k].shape)
    elif k in buffers and P + k in ini:
        pass
model.load_state_dict({**sd, **new}, strict=True)
model.eval()

if wav:
    w = wave.open(wav)
    a = np.frombuffer(w.readframes(w.getframerate() * 4), np.int16).astype(np.float32).reshape(-1, 2).T / 32768
else:
    t = np.arange(48000 * 4) / 48000
    a = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)]).astype(np.float32) * 0.3
a = a[:, : a.shape[1] // 128 * 128]
with torch.no_grad():
    est, _ = model(torch.from_numpy(a)[None])
est = est.numpy().reshape(2, -1)
s = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); ins = s.get_inputs()
st = [np.zeros(i.shape, np.float32) for i in ins[1:]]; ref = []
for h in range(a.shape[1] // 128):
    o = s.run(None, {ins[0].name: a[None, :, h * 128:(h + 1) * 128], **{i.name: v for i, v in zip(ins[1:], st)}}); st = o[1:]
    ref.append(o[0].reshape(2, 128))
ref = np.concatenate(ref, 1)
# Both carry the same one-hop output alignment; compare as-is and with a one-hop shift.
db = lambda v: 20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-20)
cands = {'aligned': (est, ref), 'torch+1 hop': (est[:, 128:], ref[:, :-128]), 'onnx+1 hop': (est[:, :-128], ref[:, 128:])}
for label, (x, y) in cands.items():
    print(f'{label:12s} torch vs onnx: {db(x - y) - db(y):7.1f} dB')
torch.save({'state_dict': model.state_dict()}, out_path)
print('saved', out_path)
