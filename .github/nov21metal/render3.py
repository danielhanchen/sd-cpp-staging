"""Correctness renders on the macOS runner's paravirtual Metal GPU (no SIMD-group ops): Z-Image with weights in RAM
(--params-backend cpu, so the matmuls the device cannot run go to the CPU) for #94, and a tiny Wan2.1 clip whose VAE
decode runs conv3d through IM2COL_3D on Metal (#98), each vs a CPU reference from the same binary."""
import argparse, hashlib, json, os, re, subprocess, time
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--bins", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
A = ap.parse_args()
OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
M = A.models
PAT = {"te": r"get_learned_condition completed, taking ([\d.]+)s", "denoise": r"sampling completed, taking ([\d.]+)s",
       "vae": r"decode_first_stage completed, taking ([\d.]+)s"}
results = {}


def run(tag, job, arm, extra=(), env_on=()):
    b = f"{A.bins}/{arm}/sd-cli"
    if job == "zimage":
        outp = OUT / f"{tag}.png"
        cmd = [b, "--mode", "img_gen", "--diffusion-model", f"{M}/z-image-turbo-Q4_K_M.gguf", "--vae", f"{M}/ae.safetensors",
               "--llm", f"{M}/Qwen3-4B-Q4_K_M.gguf", "-p", "a photo of a red fox sitting in fresh snow, golden hour",
               "--steps", "2", "-W", "256", "-H", "256", "-s", "42", "--sampling-method", "res_multistep", "--scheduler", "simple",
               "--flow-shift", "3", "--cfg-scale", "1", "--conv-direct", "off", "-o", str(outp), "-v", *extra]
    else:
        outp = OUT / tag; outp.mkdir(exist_ok=True)
        cmd = [b, "-M", "vid_gen", "--diffusion-model", f"{M}/Wan2.1-T2V-1.3B-Q4_K_M.gguf", "--vae", f"{M}/wan_2.1_vae.safetensors",
               "--t5xxl", f"{M}/umt5-xxl-encoder-Q3_K_S.gguf", "-p", "a lovely cat walking on grass", "--cfg-scale", "1.0",
               "--sampling-method", "euler", "--steps", "2", "-W", "256", "-H", "160", "--video-frames", "5", "--flow-shift", "3.0",
               "-s", "42", "--conv-direct", "off", "-o", str(outp / "f_%03d.png"), "-v", *extra]
    env = dict(os.environ)
    for k in env_on:
        env[k] = "1"
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=3600)
    text = p.stdout + p.stderr
    (OUT / f"log_{tag}.txt").write_text(" ".join(cmd) + "\n" + text[-80000:])
    files = sorted(outp.glob("*.png")) if outp.is_dir() else ([outp] if outp.exists() else [])
    r = {"tag": tag, "rc": p.returncode, "wall_s": round(time.perf_counter() - t0, 1), "frames": len(files),
         "err": [l for l in text.splitlines() if l.startswith("[E]") or "GGML_ASSERT" in l or "cannot run" in l][:4]}
    for k, pat in PAT.items():
        m = re.findall(pat, text)
        if m:
            r[k] = float(m[-1])
    r["sha"] = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16] if files else None
    results[tag] = (r, outp)
    print(json.dumps(r), flush=True)


def arr(p):
    files = sorted(p.glob("*.png")) if p.is_dir() else [p]
    return np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) for f in files])


def cmp(a, b):
    ra, pa = results[a]; rb, pb = results[b]
    if not ra["frames"] or not rb["frames"]:
        return None
    x, y = arr(pa), arr(pb)
    if x.shape != y.shape:
        return {"shape": [list(x.shape), list(y.shape)]}
    mse = float(((x - y) ** 2).mean())
    return {"max_abs": float(np.abs(x - y).max()), "mean_abs": round(float(np.abs(x - y).mean()), 3),
            "psnr": 99.0 if mse == 0 else round(10 * np.log10(255 ** 2 / mse), 2)}


run("w_cpu", "wan", "base", ["--backend", "cpu"])
run("w_base_vaecpu", "wan", "base", ["--backend", "vae=cpu"])
run("w_head_vaecpu", "wan", "head", ["--backend", "vae=cpu"])
run("w_head_vaecpu2", "wan", "head", ["--backend", "vae=cpu"])
pairs = [("w_base_vaecpu", "w_cpu"), ("w_head_vaecpu", "w_cpu"), ("w_head_vaecpu2", "w_head_vaecpu"), ("w_head_vaecpu", "w_base_vaecpu")]
summary = {"runs": {k: v[0] for k, v in results.items()}, "cmp": {f"{a}_vs_{b}": cmp(a, b) for a, b in pairs}}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary["cmp"], indent=1))
