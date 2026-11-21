"""Base vs head sd-cli renders on the macOS runner: Z-Image (rope / RMS+rope fusion, #94) and a small Wan2.1 clip (conv3d via IM2COL_3D, #98).
Arms interleaved ABBA..., one fresh process per run; writes results.jsonl + PNGs under --out."""
import argparse, hashlib, json, os, re, subprocess, time
from pathlib import Path

import numpy as np
from PIL import Image

PAT = {"load": r"loading tensors completed, taking ([\d.]+)s", "te": r"get_learned_condition completed, taking ([\d.]+)s",
       "denoise": r"sampling completed, taking ([\d.]+)s", "vae": r"decode_first_stage completed, taking ([\d.]+)s"}
ap = argparse.ArgumentParser()
ap.add_argument("--bins", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=3)
A = ap.parse_args()
OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
M = A.models
RES = OUT / "results.jsonl"


def run(job, arm, rep, env_extra=None):
    binary = f"{A.bins}/{arm}/sd-cli"
    tag = f"{job}_{arm}{'_' + '_'.join(env_extra) if env_extra else ''}_r{rep}"
    if job == "zimage":
        outp = OUT / f"{tag}.png"
        cmd = [binary, "--mode", "img_gen", "--diffusion-model", f"{M}/z-image-turbo-Q4_K_M.gguf", "--vae", f"{M}/ae.safetensors",
               "--llm", f"{M}/Qwen3-4B-Q4_K_M.gguf", "-p", "a photo of a red fox sitting in fresh snow, golden hour, detailed fur",
               "--steps", "4", "-W", "512", "-H", "512", "-s", "42", "--sampling-method", "res_multistep", "--scheduler", "simple",
               "--flow-shift", "3", "--cfg-scale", "1", "-o", str(outp), "-v"]
    else:
        fdir = OUT / tag; fdir.mkdir(exist_ok=True)
        outp = fdir
        cmd = [binary, "-M", "vid_gen", "--diffusion-model", f"{M}/Wan2.1-T2V-1.3B-Q4_K_M.gguf", "--vae", f"{M}/wan_2.1_vae.safetensors",
               "--t5xxl", f"{M}/umt5-xxl-encoder-Q3_K_S.gguf", "-p", "a lovely cat walking on grass", "--cfg-scale", "1.0",
               "--sampling-method", "euler", "--steps", "2", "-W", "480", "-H", "272", "--video-frames", "13", "--flow-shift", "3.0",
               "-s", "42", "-o", str(fdir / "f_%03d.png"), "-v"]
    env = dict(os.environ)
    for kv in env_extra or []:
        env[kv] = "1"
    t0 = time.perf_counter()
    p = subprocess.run(["/usr/bin/time", "-l", *cmd], capture_output=True, text=True, errors="replace", env=env, timeout=5400)
    wall = time.perf_counter() - t0
    text = p.stdout + p.stderr
    (OUT / f"log_{tag}.txt").write_text(" ".join(cmd) + "\n" + text[-60000:])
    r = {"job": job, "arm": arm, "env": env_extra or [], "rep": rep, "rc": p.returncode, "wall_s": round(wall, 2)}
    for k, pat in PAT.items():
        m = re.findall(pat, text)
        if m:
            r[k] = float(m[-1])
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    if m:
        r["max_rss_gb"] = round(int(m.group(1)) / 2**30, 2)
    r["has_tensor"] = "has tensor            = true" in text
    files = sorted(outp.glob("*.png")) if outp.is_dir() else ([outp] if outp.exists() else [])
    r["frames"] = len(files)
    r["sha"] = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16] if files else None
    r["path"] = str(outp)
    print(json.dumps(r), flush=True)
    with RES.open("a") as f:
        f.write(json.dumps(r) + "\n")
    return r


def arr(path):
    p = Path(path)
    files = sorted(p.glob("*.png")) if p.is_dir() else [p]
    return np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) for f in files])


def cmp(a, b):
    x, y = arr(a), arr(b)
    if x.shape != y.shape:
        return {"shape_mismatch": [list(x.shape), list(y.shape)]}
    mse = float(((x - y) ** 2).mean())
    return {"max_abs": float(np.abs(x - y).max()), "psnr": (99.0 if mse == 0 else round(10 * np.log10(255 ** 2 / mse), 2))}


order = ["base", "head", "head", "base", "base", "head", "head", "base"][: 2 * A.reps]
runs = []
for job in ("zimage", "wan"):
    reps = {"base": 0, "head": 0}
    for arm in order if job == "zimage" else ["base", "head", "head", "base"]:
        runs.append(run(job, arm, reps[arm])); reps[arm] += 1
    # #94 attribution: head with fusion off (sd.rope_apply only), and base A/A
    if job == "zimage":
        runs.append(run(job, "head", 0, ["GGML_METAL_FUSION_DISABLE"]))
summary = {}
for job in ("zimage", "wan"):
    rs = [r for r in runs if r["job"] == job and r["rc"] == 0]
    b = [r for r in rs if r["arm"] == "base" and not r["env"]]
    h = [r for r in rs if r["arm"] == "head" and not r["env"]]
    hf = [r for r in rs if r["arm"] == "head" and r["env"]]
    s = {}
    for k in ("denoise", "vae", "wall_s", "max_rss_gb"):
        for name, grp in (("base", b), ("head", h)):
            v = sorted(r[k] for r in grp if k in r)
            if v:
                s[f"{name}_{k}"] = {"median": v[len(v) // 2], "min": v[0], "max": v[-1], "n": len(v)}
    if len(b) >= 2:
        s["base_AA"] = cmp(b[0]["path"], b[1]["path"])
    if len(h) >= 2:
        s["head_AA"] = cmp(h[0]["path"], h[1]["path"])
    if b and h:
        s["head_vs_base"] = cmp(b[0]["path"], h[0]["path"])
    if hf and h:
        s["head_fusionoff_vs_head"] = cmp(h[0]["path"], hf[0]["path"])
    if hf and b:
        s["head_fusionoff_vs_base"] = cmp(b[0]["path"], hf[0]["path"])
    summary[job] = s
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
