#!/usr/bin/env python3
"""Fine-tune the StemgenRT vocal model for the phone: 48 kHz audio, fp16-robust backbone.

Runs unattended on a Colab VM and survives losing the laptop or the VM:
  * every phase is idempotent; a restart picks up where the last one stopped;
  * the full training state (raw weights, EMA, Adam, RNG, step, loss scale) is
    written atomically to Google Drive every CKPT_EVERY updates;
  * status.json (Drive + local) always says what is happening and how long is left.

The update itself is StemgenRT's own `grouped_update` (its objectives, Adam,
gradient clipping and parameter EMA). Two changes for the phone:
  * audio is MUSDB18-HQ resampled to 48 kHz (the phone's rate; the release was
    trained at 44.1 kHz);
  * the recurrent backbone runs under fp16 autocast (the Hexagon NPU computes in
    fp16), with dynamic loss scaling; FFT front/back end stay fp32 as on the phone.

usage: python train_phone.py --src STEMGEN_SRC --init vocals_released.pt [--steps 8000]
       python train_phone.py --status        (print status and exit)
"""
import argparse
import io
import json
import math
import os
import random
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

WORK = os.environ.get("STEMGEN_WORK", "/content/work")
DATA = os.environ.get("STEMGEN_DATA", "/content/data")
DRIVE_FOLDER = "stemgen_phone"
STATUS_LOCAL = f"{WORK}/status.json"
SR = 48000
STEMS = ("drums", "bass", "vocals", "other")          # StemgenRT source order
CROP = 264576                                         # 88064 warmup + 176512 scored
WARMUP = 88064
BATCH = int(os.environ.get("STEMGEN_BATCH", "16"))
CKPT_EVERY = int(os.environ.get("STEMGEN_CKPT_EVERY", "100"))
MUSDB_URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"


# ----------------------------------------------------------------- Drive ----

class Drive:
    """Minimal Google Drive store (after `colab auth`): one folder, files by name."""

    def __init__(self):
        import google.auth
        from googleapiclient.discovery import build
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
        self.svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        q = (f"name='{DRIVE_FOLDER}' and mimeType='application/vnd.google-apps.folder' "
             "and trashed=false and 'root' in parents")
        found = self.svc.files().list(q=q, fields="files(id)").execute()["files"]
        if found:
            self.folder = found[0]["id"]
        else:
            meta = {"name": DRIVE_FOLDER, "mimeType": "application/vnd.google-apps.folder"}
            self.folder = self.svc.files().create(body=meta, fields="id").execute()["id"]

    def _find(self, name):
        q = f"name='{name}' and '{self.folder}' in parents and trashed=false"
        files = self.svc.files().list(q=q, fields="files(id,size,modifiedTime)").execute()["files"]
        return files[0] if files else None

    def put(self, local, name):
        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(local, resumable=True, chunksize=32 * 2**20)
        existing = self._find(name)
        if existing:
            self.svc.files().update(fileId=existing["id"], media_body=media).execute()
        else:
            self.svc.files().create(body={"name": name, "parents": [self.folder]},
                                    media_body=media, fields="id").execute()

    def get(self, name, local):
        from googleapiclient.http import MediaIoBaseDownload
        f = self._find(name)
        if not f:
            return False
        tmp = local + ".part"
        with io.FileIO(tmp, "wb") as fh:
            dl = MediaIoBaseDownload(fh, self.svc.files().get_media(fileId=f["id"]), chunksize=32 * 2**20)
            done = False
            while not done:
                _, done = dl.next_chunk()
        os.replace(tmp, local)
        return True


class LocalStore:
    """Drive stand-in for local smoke tests (STEMGEN_STORE=dir)."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def put(self, local, name):
        import shutil
        shutil.copyfile(local, f"{self.root}/{name}.tmp")
        os.replace(f"{self.root}/{name}.tmp", f"{self.root}/{name}")

    def get(self, name, local):
        import shutil
        if not os.path.exists(f"{self.root}/{name}"):
            return False
        shutil.copyfile(f"{self.root}/{name}", local)
        return True


def save_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


class Status:
    def __init__(self, drive):
        self.drive, self.data, self.last_push = drive, {}, 0.0

    def set(self, push=False, **kw):
        self.data.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        save_json_atomic(STATUS_LOCAL, self.data)
        if push or time.time() - self.last_push > 120:
            try:
                self.drive.put(STATUS_LOCAL, "status.json")
                self.last_push = time.time()
            except Exception as e:                      # status is best effort
                print("status upload failed:", e, flush=True)
        line = " | ".join(f"{k}={v}" for k, v in self.data.items() if k not in ("history",))
        print(line, flush=True)


def fmt_eta(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


# ------------------------------------------------------------------ data ----

def prepare_data(status):
    """Download MUSDB18-HQ and cache every stem resampled to 48 kHz (int16 .npy)."""
    done_flag = f"{DATA}/48k/.done"
    if os.path.exists(done_flag):
        return
    import soundfile as sf
    import soxr
    os.makedirs(DATA, exist_ok=True)
    zip_path = f"{DATA}/musdb18hq.zip"
    if not os.path.exists(f"{DATA}/musdb18hq/.unzipped"):
        status.set(phase="download MUSDB18-HQ (22.6 GB)", push=True)
        subprocess.run(["wget", "-q", "-c", "-O", zip_path, MUSDB_URL], check=True)
        status.set(phase="unzip MUSDB18-HQ", push=True)
        subprocess.run(["unzip", "-q", "-o", zip_path, "-d", f"{DATA}/musdb18hq"], check=True)
        open(f"{DATA}/musdb18hq/.unzipped", "w").close()
        os.remove(zip_path)
    tracks = [(split, t) for split in ("train", "test")
              for t in sorted(os.listdir(f"{DATA}/musdb18hq/{split}"))]
    t0 = time.time()
    for i, (split, track) in enumerate(tracks):
        out = f"{DATA}/48k/{split}/{track}"
        if os.path.exists(f"{out}/.done"):
            continue
        os.makedirs(out, exist_ok=True)
        for stem in STEMS:
            audio, rate = sf.read(f"{DATA}/musdb18hq/{split}/{track}/{stem}.wav", dtype="float32")
            assert rate == 44100 and audio.ndim == 2 and audio.shape[1] == 2, (track, stem, rate)
            res = soxr.resample(audio, 44100, SR, quality="VHQ")
            np.save(f"{out}/{stem}.npy", np.clip(np.round(res.T * 32767), -32768, 32767).astype(np.int16))
        open(f"{out}/.done", "w").close()
        rate_s = (time.time() - t0) / (i + 1)
        status.set(phase=f"resample to 48 kHz {i + 1}/{len(tracks)}", eta_phase=fmt_eta(rate_s * (len(tracks) - i - 1)))
    open(done_flag, "w").close()


class Crops:
    """Random training crops: uniform track, vocal-active with p=0.85, gain/swap augmentation."""

    def __init__(self, split, rng):
        root = f"{DATA}/48k/{split}"
        self.tracks = []
        for t in sorted(os.listdir(root)):
            stems = [np.load(f"{root}/{t}/{s}.npy", mmap_mode="r") for s in STEMS]
            if stems[0].shape[1] > CROP:
                self.tracks.append(stems)
        self.rng = rng

    def batch(self):
        import torch
        mix = np.zeros((BATCH, 2, CROP), np.float32)
        tgt = np.zeros((BATCH, 4, 2, CROP), np.float32)
        for b in range(BATCH):
            want_vocals = self.rng.random() < 0.85
            for _ in range(20):
                stems = self.tracks[self.rng.randrange(len(self.tracks))]
                start = self.rng.randrange(stems[0].shape[1] - CROP)
                crop = np.stack([s[:, start:start + CROP] for s in stems]).astype(np.float32) / 32767
                scored = crop[2, :, WARMUP:]
                if not want_vocals or np.sqrt(np.mean(scored ** 2)) > 3e-3:
                    break
            gains = np.array([self.rng.uniform(0.7, 1.3) for _ in STEMS], np.float32)[:, None, None]
            crop *= gains
            if self.rng.random() < 0.5:
                crop = crop[:, ::-1]
            tgt[b] = crop
            mix[b] = crop.sum(0)
        return torch.from_numpy(mix), torch.from_numpy(tgt)


# ------------------------------------------------------------------ model ----

def build_model(src, init_path, device):
    import torch
    sys.path.insert(0, src)
    from stemgenrt.banded import BandSeparator
    model = BandSeparator(sources=("vocals",), band_width=96, global_width=192, layers=2)
    state = torch.load(init_path, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device)


def fp16_backbone(model, enabled):
    """Run the recurrent backbone under fp16 autocast, like the NPU; outputs back to fp32."""
    import torch
    original = model.backbone.forward

    def forward(*args, **kwargs):
        if not enabled[0]:
            return original(*args, **kwargs)
        with torch.autocast("cuda", dtype=torch.float16):
            out = original(*args, **kwargs)
        return tuple(o.float() for o in out)

    model.backbone.forward = forward


class LossScale:
    """Dynamic loss scaling around StemgenRT's output-derivative replay."""

    def __init__(self, model, value=1024.0):
        import torch
        self.value = value
        self.good = 0
        original = torch.autograd.backward
        scale = self

        def backward(tensors, grad_tensors=None, *args, **kwargs):
            if grad_tensors is not None:
                grad_tensors = tuple(g * scale.value for g in grad_tensors)
            return original(tensors, grad_tensors, *args, **kwargs)

        torch.autograd.backward = backward
        for p in model.parameters():
            p.register_hook(lambda g: g / scale.value)

    def ok(self):
        self.good += 1
        if self.good >= 500 and self.value < 2 ** 16:
            self.value *= 2
            self.good = 0

    def overflow(self):
        self.value = max(1.0, self.value / 2)
        self.good = 0


def lr_at(step, total, peak=3e-5, floor=3e-6, warmup=100):
    if step <= warmup:
        return peak * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(1.0, t)))


# ------------------------------------------------------------- evaluation ----

def evaluate(src, weights, device, fp16, limit=0):
    """Global vocal SDR on 30 s of each MUSDB18-HQ test track (48 kHz)."""
    import torch
    sys.path.insert(0, src)
    from stemgenrt.banded import BandSeparator
    model = BandSeparator(sources=("vocals",), band_width=96, global_width=192, layers=2)
    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()
    if fp16:
        fp16_backbone(model, [True])
    root = f"{DATA}/48k/test"
    sdrs = []
    for t in sorted(os.listdir(root))[: limit or None]:
        stems = [np.load(f"{root}/{t}/{s}.npy", mmap_mode="r") for s in STEMS]
        n = stems[0].shape[1]
        start, length = min(60 * SR, max(0, n - 30 * SR)), 30 * SR // 128 * 128
        if n < start + length:
            continue
        crop = np.stack([s[:, start:start + length] for s in stems]).astype(np.float32) / 32767
        mix = torch.from_numpy(crop.sum(0))[None].to(device)
        with torch.no_grad():
            est, _ = model(mix)
        est = est.reshape(2, -1).cpu().numpy()
        ref = crop[2]
        if np.mean(ref ** 2) < 1e-6:
            continue
        sdrs.append(10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + 1e-12)))
    return float(np.median(sdrs)), float(np.mean(sdrs)), len(sdrs)


# ------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--init")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--eval-tracks", type=int, default=0, help="limit evaluation (smoke tests)")
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    if args.src:
        sys.path.insert(0, args.src)
    drive = LocalStore(os.environ["STEMGEN_STORE"]) if os.environ.get("STEMGEN_STORE") else Drive()
    if args.status:
        if not os.path.exists(STATUS_LOCAL):
            drive.get("status.json", STATUS_LOCAL)
        print(open(STATUS_LOCAL).read() if os.path.exists(STATUS_LOCAL) else "no status yet")
        return

    status = Status(drive)
    status.set(phase="setup", total_steps=args.steps, push=True)
    prepare_data(status)

    import torch
    device = args.device
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = build_model(args.src, args.init, device)
    from stemgenrt.checkpoint import ParameterEMA
    from stemgenrt.losses import grouped_update
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ckpt_local = f"{WORK}/ckpt.pt"
    step, history = 0, []
    rng = random.Random(20261123)
    resumed = drive.get("ckpt.pt", ckpt_local)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, foreach=False)
    if resumed:
        ck = torch.load(ckpt_local, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        step, history = ck["step"], ck.get("history", [])
        ema = ParameterEMA.from_state_dict(model, ck["ema"], expected_step=step)
        rng.setstate(ck["py_rng"])
        torch.set_rng_state(ck["torch_rng"])
        scale_value = ck["loss_scale"]
    else:
        ema = ParameterEMA(model, decay=0.995)
        scale_value = 1024.0
    model.train()
    enabled = [not args.no_fp16]
    fp16_backbone(model, enabled)
    scale = LossScale(model, scale_value)
    crops = Crops("train", rng)
    status.set(phase="train", step=step, resumed_from=step if resumed else 0, tracks=len(crops.tracks), push=True)

    t_start, s_start = time.time(), step
    losses = []
    while step < args.steps:
        mix, tgt = crops.batch()
        lr = lr_at(step + 1, args.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        for attempt in range(6):
            enabled[0] = attempt < 5 and not args.no_fp16  # last resort: one fp32 step
            try:
                result = grouped_update(model, optimizer, ema, mix, tgt, step=step + 1,
                                        warmup_samples=WARMUP, ordinary_microbatch=4,
                                        auxiliary_microbatch=2, target_source="vocals",
                                        extra_ordinary_primary_sdr_weight=0.2)
                scale.ok()
                break
            except Exception as e:                       # non-finite fp16 gradients
                if "finite" not in str(e).lower():
                    raise
                print(f"step {step + 1}: overflow at loss scale {scale.value}, retrying", flush=True)
                scale.overflow()
        else:
            raise RuntimeError("update failed even in fp32")
        enabled[0] = not args.no_fp16
        step += 1
        losses.append(float(result["weighted_loss"]))
        rate = (time.time() - t_start) / (step - s_start)
        if step % 10 == 0 or step == args.steps:
            status.set(phase="train", step=step, total_steps=args.steps,
                       percent=round(100 * step / args.steps, 1), sec_per_step=round(rate, 2),
                       eta=fmt_eta(rate * (args.steps - step)), lr=f"{lr:.2e}",
                       loss_last100=round(float(np.mean(losses[-100:])), 4), loss_scale=scale.value)
        if step % CKPT_EVERY == 0 or step == args.steps:
            history.append({"step": step, "loss": float(np.mean(losses[-CKPT_EVERY:]))})
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "ema": ema.state_dict(model), "step": step, "history": history,
                        "py_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                        "loss_scale": scale.value}, ckpt_local + ".tmp")
            os.replace(ckpt_local + ".tmp", ckpt_local)
            drive.put(ckpt_local, "ckpt.pt")
            status.set(last_checkpoint=step, push=True)

    # Final EMA weights and a quality check against the released model at 48 kHz.
    ema_state = {**ema.state_dict(model)["parameters"], **ema.state_dict(model)["buffers"]}
    torch.save({"state_dict": ema_state}, f"{WORK}/vocals_phone_ema.pt")
    drive.put(f"{WORK}/vocals_phone_ema.pt", "vocals_phone_ema.pt")
    status.set(phase="evaluate on MUSDB18-HQ test (48 kHz)", push=True)
    results = {}
    for label, weights in (("released", base_state), ("finetuned", ema_state)):
        for fp16 in ((False,) if args.no_fp16 else (False, True)):
            med, mean, n = evaluate(args.src, weights, device, fp16, args.eval_tracks)
            results[f"{label}_{'fp16' if fp16 else 'fp32'}"] = {"median_sdr": round(med, 3), "mean_sdr": round(mean, 3), "tracks": n}
            status.set(results=results)
    save_json_atomic(f"{WORK}/results.json", results)
    drive.put(f"{WORK}/results.json", "results.json")
    status.set(phase="done", results=results, push=True)


if __name__ == "__main__":
    main()
