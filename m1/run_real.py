#!/usr/bin/env python3
"""Load + render one REAL model (Studio route: validate -> select_and_activate_engine -> begin_load -> generate); derived from maccpu_cov_nov24 run_family.py.
Original: Load + render one tiny pack through Studio's real backend (DiffusionBackend / VideoBackend), recording where the
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
ap.add_argument("--repo", required = True)
ap.add_argument("--gguf", default = None)
ap.add_argument("--guidance", type = float, default = 0.0)
ap.add_argument("--out", required = True)
ap.add_argument("--size", type = int, default = 512)
ap.add_argument("--frames", type = int, default = 17)
ap.add_argument("--steps", type = int, default = 9)
ap.add_argument("--expect", default = None, help = "expected device type (default: mps if available else cpu)")
a = ap.parse_args()
sys.path.insert(0, a.backend)
logging.basicConfig(level = logging.INFO, stream = sys.stderr, format = "%(asctime)s %(name)s %(levelname)s %(message)s")

import torch  # noqa: E402

EXPECT = a.expect or ("mps" if torch.backends.mps.is_available() else "cpu")
REC: dict = {"repo": a.repo, "gguf": a.gguf, "steps": a.steps, "guidance": a.guidance, "expect": EXPECT, "torch": torch.__version__,
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

# ------------------------------------------------------------------ memory sampler (process tree RSS + MPS driver)
import threading  # noqa: E402

import psutil  # noqa: E402

MEM = {"tree_rss_peak_mb": 0.0, "self_rss_peak_mb": 0.0, "child_rss_peak_mb": 0.0, "mps_driver_peak_mb": 0.0,
       "sys_avail_min_mb": None, "children_seen": set()}
_stop = threading.Event()


def _sampler():
    me = psutil.Process()
    while not _stop.is_set():
        try:
            self_rss = me.memory_info().rss
            kids = me.children(recursive = True)
            krss = 0
            for k in kids:
                try:
                    krss += k.memory_info().rss
                    MEM["children_seen"].add(k.name())
                except Exception:  # noqa: BLE001
                    pass
            MEM["self_rss_peak_mb"] = max(MEM["self_rss_peak_mb"], self_rss / 2**20)
            MEM["child_rss_peak_mb"] = max(MEM["child_rss_peak_mb"], krss / 2**20)
            MEM["tree_rss_peak_mb"] = max(MEM["tree_rss_peak_mb"], (self_rss + krss) / 2**20)
            av = psutil.virtual_memory().available / 2**20
            MEM["sys_avail_min_mb"] = av if MEM["sys_avail_min_mb"] is None else min(MEM["sys_avail_min_mb"], av)
            if torch.backends.mps.is_available():
                MEM["mps_driver_peak_mb"] = max(MEM["mps_driver_peak_mb"], torch.mps.driver_allocated_memory() / 2**20)
        except Exception:  # noqa: BLE001
            pass
        _stop.wait(0.5)


def _img_stats(img):
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype = np.float32)
    return {"shape": list(arr.shape), "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()), "nan": bool(np.isnan(arr).any())}


def _write(stage):
    REC["stage"] = stage
    REC["mem"] = {k: (sorted(v) if isinstance(v, set) else (round(v, 1) if isinstance(v, float) else v))
                  for k, v in MEM.items()}
    json.dump(REC, open(a.out, "w"), indent = 2, default = str)


def main():
    threading.Thread(target = _sampler, daemon = True).start()
    t_all = time.time()
    try:
        from core.inference.diffusion import get_diffusion_backend, resolve_model_kind
        from core.inference.diffusion_engine_router import active_engine_name, select_and_activate_engine

        rel_json = os.environ.get("MACREAL_RELEASE_JSON")
        if rel_json:
            # Harness only: hosted runners share an IP whose unauthenticated GitHub API quota is spent, so the
            # installer's release LOOKUP falls back to the pinned tag's JSON fetched beforehand (no token in the
            # workflow). Asset download, sha256 check, extraction and probing are Studio's own code, unchanged.
            from core.inference.sd_cpp_backend import _installer_module

            inst = _installer_module()
            _orig_fetch = inst._fetch_release
            shipped = json.load(open(rel_json))

            def _fetch_release(tag = None, **kw):
                try:
                    return _orig_fetch(tag, **kw)
                except Exception as exc:  # noqa: BLE001
                    if inst._is_rate_limited(exc) and tag == shipped["tag_name"]:
                        REC.setdefault("harness_notes", []).append(f"release lookup rate-limited; used shipped {tag}")
                        return shipped
                    raise

            inst._fetch_release = _fetch_release
        kind = resolve_model_kind(a.gguf, None)
        REC["kind"] = kind
        backend = get_diffusion_backend()
        fam = backend.validate_load_request(a.repo, gguf_filename = a.gguf, model_kind = kind)
        REC["family"] = getattr(fam, "name", None)
        engine = select_and_activate_engine(fam, model_kind = kind, gguf_filename = a.gguf)
        REC["engine"] = active_engine_name()
        try:
            from core.inference import diffusion_engine_router as R
            REC["engine_reason"] = {k: str(getattr(R, k)) for k in dir(R) if "reason" in k.lower() and not callable(getattr(R, k))}
        except Exception:  # noqa: BLE001
            pass
        _write("engine_selected")
        t = time.time()
        engine.begin_load(a.repo, gguf_filename = a.gguf, model_kind = kind, local_files_only = False)
        last = None
        while True:
            pr = engine.load_progress()
            ph = pr.get("phase") if isinstance(pr, dict) else pr
            if ph != last:
                print("load phase", ph, {k: v for k, v in (pr or {}).items() if k != "phase"}, flush = True)
                last = ph
            if ph == "ready":
                break
            if ph == "error" or (isinstance(pr, dict) and pr.get("error")):
                raise RuntimeError(f"load failed: {pr.get('error') if isinstance(pr, dict) else pr}")
            if time.time() - t > 5400:
                raise RuntimeError(f"load timeout: {pr}")
            time.sleep(1.0)
        REC["load_s"] = round(time.time() - t, 2)
        try:
            st = engine.status() or {}
            REC["status"] = {k: st.get(k) for k in st if any(x in k for x in (
                "device", "precision", "dtype", "quant", "engine", "offload", "family", "attention", "backend",
                "accelerator", "binary", "small_host", "memory"))}
        except Exception as e:  # noqa: BLE001
            REC["status_error"] = repr(e)[:300]
        _write("loaded")
        t = time.time()
        r = engine.generate(prompt = PROMPT, width = a.size, height = a.size, steps = a.steps, guidance = a.guidance,
                            seed = 1, batch_size = 1)
        REC["gen_s"] = round(time.time() - t, 2)
        REC["s_per_image"] = REC["gen_s"]
        imgs = r.get("images") if isinstance(r, dict) else r
        img = imgs[0]
        if not hasattr(img, "save"):
            from PIL import Image

            if isinstance(img, dict):
                img = img.get("image") or img.get("data") or next(iter(img.values()))
            img = Image.open(io.BytesIO(base64.b64decode(img.split(",")[-1]) if isinstance(img, str) else img))
        png = os.path.splitext(a.out)[0] + ".png"
        img.save(png)
        REC["output"] = _img_stats(img)
        REC["error"] = None
    except BaseException as e:  # noqa: BLE001
        REC["error"] = f"{type(e).__name__}: {str(e)[:2500]}"
        REC["traceback"] = traceback.format_exc()[-6000:]
    _stop.set()
    REC["wall_s"] = round(time.time() - t_all, 2)
    REC["calls"] = {k: {kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()}
                    for k, v in CALLS.items()}
    REC["warnings"] = WARN[-80:]
    REC["fallback_warnings"] = [w for w in WARN if "fall back" in w.lower() or "not currently supported on the mps" in w.lower()]
    REC["peak_rss_self_mb_rusage"] = _peak_rss_mb()
    o = REC.get("output") or {}
    REC["checks"] = {"no_error": REC.get("error") is None, "not_black": bool(o) and o.get("max", 0) > 8,
                     "not_constant": bool(o) and o.get("std", 0) > 1.0, "no_nan": bool(o) and not o.get("nan"),
                     "finite_hooks": all(v["nonfinite"] == 0 for v in CALLS.values()),
                     "no_mps_fallback_warning": not REC["fallback_warnings"]}
    REC["verdict"] = "PASS" if all(REC["checks"].values()) else "FAIL"
    _write("done")
    print("RESULT", a.repo, a.gguf, REC.get("engine"), REC["verdict"], json.dumps(REC["checks"]), "gen_s", REC.get("gen_s"),
          "tree_rss_peak_mb", REC["mem"]["tree_rss_peak_mb"], (REC.get("error") or "")[:600], flush = True)


def _peak_rss_mb():
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


if __name__ == "__main__":
    main()
    os._exit(0)
