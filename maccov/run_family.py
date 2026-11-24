#!/usr/bin/env python3
"""Load + render one tiny pack through Studio's real backend (DiffusionBackend / VideoBackend), recording where the
denoiser and the VAE decoder actually ran and whether their outputs are finite.

  run_family.py --backend STUDIO_BACKEND --pack DIR --family FAM --kind image|video --out OUT_JSON
                [--size 512] [--frames 17] [--steps 2]

Checks (verdict PASS only if all hold):
  * load + generate return without raising
  * every denoiser / VAE-decoder forward ran on the expected device (mps on a Mac; cpu with --expect cpu)
  * every recorded denoiser / decoder output is finite
  * the returned image / frames are not constant and not all black
  * no "fall back to run on the CPU" / MPS-unsupported warning in this process
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
import traceback
import warnings

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required = True)
ap.add_argument("--pack", required = True)
ap.add_argument("--family", required = True)
ap.add_argument("--kind", choices = ("image", "video"), required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--size", type = int, default = 512)
ap.add_argument("--frames", type = int, default = 17)
ap.add_argument("--steps", type = int, default = 2)
ap.add_argument("--expect", default = None, help = "expected device type (default: mps if available else cpu)")
a = ap.parse_args()
sys.path.insert(0, a.backend)
logging.basicConfig(level = logging.INFO, stream = sys.stderr, format = "%(asctime)s %(name)s %(levelname)s %(message)s")

import torch  # noqa: E402

EXPECT = a.expect or ("mps" if torch.backends.mps.is_available() else "cpu")
REC: dict = {"family": a.family, "kind": a.kind, "pack": a.pack, "expect": EXPECT, "torch": torch.__version__,
             "mps_available": torch.backends.mps.is_available(),
             "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}
WARN: list = []
_orig_showwarning = warnings.showwarning


def _show(message, category, filename, lineno, file = None, line = None):
    WARN.append(f"{category.__name__}: {message} ({os.path.basename(filename)}:{lineno})")
    _orig_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _show
warnings.simplefilter("always")


class _LogTap(logging.Handler):
    def emit(self, r):
        msg = r.getMessage()
        low = msg.lower()
        if "fall back" in low or "fallback" in low or "not currently supported on the mps" in low or "mps" in low:
            WARN.append(f"LOG {r.name}: {msg[:400]}")


logging.getLogger().addHandler(_LogTap())

# --------------------------------------------------------------------------------------------- forward device taps
CALLS: dict = {}


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for y in x:
            t = _first_tensor(y)
            if t is not None:
                return t
    if isinstance(x, dict):
        for y in x.values():
            t = _first_tensor(y)
            if t is not None:
                return t
    if hasattr(x, "to_tuple"):
        try:
            return _first_tensor(x.to_tuple())
        except Exception:  # noqa: BLE001
            return None
    if hasattr(x, "sample") and isinstance(getattr(x, "sample"), torch.Tensor):
        return x.sample
    return None


def _role(m):
    n = type(m).__name__
    if "Transformer" in n and ("2DModel" in n or "3DModel" in n or n.endswith("Transformer")):
        return "denoiser"
    if n in ("Decoder",) or n.endswith("Decoder3d") or n.endswith("Decoder3D") or n.endswith("VideoDecoder3d"):
        return "vae_decoder"
    if n.startswith("AutoencoderKL"):
        return "vae"
    if "Vocoder" in n:
        return "vocoder"
    if "Connector" in n:
        return "connectors"
    if "Upsampler" in n:
        return "latent_upsampler"
    return None


def _hook(m, inp, kw, out):
    r = _role(m)
    if r is None:
        return
    key = f"{r}:{type(m).__name__}"
    d = CALLS.setdefault(key, {"calls": 0, "in_dev": set(), "out_dev": set(), "param_dev": set(), "nonfinite": 0,
                               "dtypes": set()})
    d["calls"] += 1
    ti = _first_tensor(inp)
    if ti is None:
        ti = _first_tensor(kw)
    to = _first_tensor(out)
    if ti is not None:
        d["in_dev"].add(ti.device.type)
    if to is not None:
        d["out_dev"].add(to.device.type)
        d["dtypes"].add(str(to.dtype).replace("torch.", ""))
        if to.is_floating_point() and not bool(torch.isfinite(to.detach()).all()):
            d["nonfinite"] += 1
    p = next(m.parameters(), None)
    if p is not None:
        d["param_dev"].add(p.device.type)


torch.nn.modules.module.register_module_forward_hook(_hook, with_kwargs = True)

PROMPT = "a red fox sitting in fresh snow, golden hour, detailed photograph"


def _img_stats(img):
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype = np.float32)
    return {"shape": list(arr.shape), "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


def route_info():
    """What Studio's router would pick on this host for the real repo (no load)."""
    info = {}
    try:
        from core.inference.diffusion_engine_router import mlx_mode, mlx_selection
        from core.inference.diffusion_families import detect_family

        src = json.load(open(os.path.join(a.pack, "_tiny_source.json")))["repo"]
        fam = detect_family(src)
        info["real_repo"] = src
        info["detect_family"] = getattr(fam, "name", None)
        info["mlx_mode"] = mlx_mode()
        info["mlx_selection_diffusers"] = list(mlx_selection(fam, model_kind = None, gguf_filename = None))
        info["mlx_selection_gguf"] = list(mlx_selection(fam, model_kind = "gguf", gguf_filename = "x-Q4_K_M.gguf"))
    except Exception as e:  # noqa: BLE001
        info["error"] = repr(e)[:400]
    return info


def main():
    t_all = time.time()
    if a.kind == "image":
        REC["route"] = route_info()
    try:
        if a.kind == "image":
            from core.inference.diffusion import DiffusionBackend

            b = DiffusionBackend()
            t = time.time()
            b.load_pipeline(a.pack, family_override = a.family, local_files_only = True)
            REC["load_s"] = round(time.time() - t, 2)
            try:
                st = b.status() or {}
                REC["status"] = {k: st.get(k) for k in st if any(x in k for x in (
                    "device", "precision", "dtype", "quant", "engine", "offload", "family", "attention", "backend"))}
            except Exception as e:  # noqa: BLE001
                REC["status_error"] = repr(e)[:300]
            t = time.time()
            r = b.generate(prompt = PROMPT, width = a.size, height = a.size, steps = a.steps, seed = 1, batch_size = 1,
                           live_preview = False)
            REC["gen_s"] = round(time.time() - t, 2)
            imgs = r.get("images") if isinstance(r, dict) else r
            img = imgs[0]
            if not hasattr(img, "save"):
                from PIL import Image

                img = Image.open(io.BytesIO(base64.b64decode(img.split(",")[-1]) if isinstance(img, str) else img))
            png = os.path.splitext(a.out)[0] + ".png"
            img.save(png)
            REC["output"] = _img_stats(img)
        else:
            from core.inference.video import get_video_backend

            vb = get_video_backend()
            t = time.time()
            vb.begin_load(a.pack, family_override = a.family, local_files_only = True)
            while True:
                pr = vb.load_progress()
                ph = pr.get("phase") if isinstance(pr, dict) else pr
                if ph == "ready":
                    break
                if ph in ("error", "failed") or (isinstance(pr, dict) and pr.get("error")):
                    raise RuntimeError(f"load failed: {pr}")
                if time.time() - t > 1800:
                    raise RuntimeError(f"load timeout: {pr}")
                time.sleep(0.5)
            REC["load_s"] = round(time.time() - t, 2)
            try:
                st = vb.status() or {}
                REC["status"] = {k: st.get(k) for k in st if any(x in k for x in (
                    "device", "precision", "dtype", "quant", "offload", "family", "attention", "defaults"))}
            except Exception as e:  # noqa: BLE001
                REC["status_error"] = repr(e)[:300]
            t = time.time()
            r = vb.generate(prompt = PROMPT, width = a.size, height = a.size, num_frames = a.frames, seed = 1,
                            steps = a.steps, live_preview = False)
            REC["gen_s"] = round(time.time() - t, 2)
            REC["video_meta"] = {k: r.get(k) for k in ("width", "height", "num_frames", "fps", "steps", "has_audio")}
            mp4 = os.path.splitext(a.out)[0] + ".mp4"
            with open(mp4, "wb") as fh:
                fh.write(r["mp4_bytes"])
            import av
            import numpy as np

            with av.open(mp4) as c:
                frames = [f.to_ndarray(format = "rgb24").astype(np.float32) for f in c.decode(c.streams.video[0])]
                REC["mp4_audio_streams"] = len(c.streams.audio)
            arr = np.stack(frames)
            REC["output"] = {"frames": len(frames), "shape": list(arr.shape), "mean": float(arr.mean()),
                             "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max()),
                             "per_frame_std_min": float(min(f.std() for f in frames))}
        REC["error"] = None
    except BaseException as e:  # noqa: BLE001
        REC["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
        REC["traceback"] = traceback.format_exc()[-6000:]
    REC["wall_s"] = round(time.time() - t_all, 2)
    REC["calls"] = {k: {kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()}
                    for k, v in CALLS.items()}
    REC["warnings"] = WARN[-60:]
    REC["peak_rss_mb"] = _peak_rss_mb()
    REC["checks"], REC["verdict"] = verdict()
    json.dump(REC, open(a.out, "w"), indent = 2, default = str)
    print("RESULT", a.family, REC["verdict"], json.dumps(REC["checks"]), REC.get("error") or "", flush = True)


def _peak_rss_mb():
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


def verdict():
    c = {}
    c["no_error"] = REC.get("error") is None
    den = [v for k, v in CALLS.items() if k.startswith("denoiser:")]
    dec = [v for k, v in CALLS.items() if k.startswith(("vae_decoder:", "vae:"))]
    c["denoiser_ran"] = bool(den) and all(v["calls"] > 0 for v in den)
    c["denoiser_on_expected"] = bool(den) and all(v["in_dev"] <= {EXPECT} and v["param_dev"] <= {EXPECT} for v in den)
    c["decoder_ran"] = bool(dec)
    c["decoder_on_expected"] = bool(dec) and all(v["in_dev"] <= {EXPECT} for v in dec)
    c["finite"] = all(v["nonfinite"] == 0 for v in CALLS.values())
    o = REC.get("output") or {}
    c["not_constant"] = bool(o) and o.get("std", 0) > 1.0
    c["not_black"] = bool(o) and o.get("max", 0) > 8
    bad = [w for w in WARN if "fall back" in w.lower() or "not currently supported on the mps" in w.lower()]
    c["no_mps_fallback_warning"] = not bad
    return c, ("PASS" if all(c.values()) else "FAIL")


if __name__ == "__main__":
    main()
    os._exit(0)
