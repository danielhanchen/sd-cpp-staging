#!/usr/bin/env python3
"""S03 VAE-decode A/B on one macOS runner: fresh sd-cli process per run, arms rotated per repeat.

usage: bench.py --bin-dir bin --model models/sd_turbo-f16.gguf --out results --res 1024 --reps 3 [--slow-arm ...]
Arms (all Metal, GGML_METAL_FORCE_SIMDGROUP=1, Studio's darwin argv):
  base_im2col  base build, --conv-direct off         (today's Studio)
  head_im2col  head build, --conv-direct off         (A/A: same code path, must be pixel identical)
  head_direct  head build, --conv-direct off,vae=on  (implicit GEMM VAE)
  base_direct  base build, --conv-direct off,vae=on  (scalar direct kernel; once, with a timeout)
PSNR of every image vs the first base_im2col image.
"""
import argparse, hashlib, json, math, os, re, statistics as st, subprocess, time
from pathlib import Path

import numpy as np
from PIL import Image

PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
STUDIO = ["--clip-on-cpu", "--mmap", "-t", "3", "--diffusion-fa"]
ARMS = {
    "base_im2col": ("base", "off"),
    "head_im2col": ("head", "off"),
    "head_direct": ("head", "off,vae=on"),
    "base_direct": ("base", "off,vae=on"),
}


def grab(pat, text):
    m = re.findall(pat, text)
    return float(m[-1]) if m else None


def run(arm, a, out, tag, timeout):
    build, mode = ARMS[arm]
    img = out / f"{tag}.png"
    cmd = ["/usr/bin/time", "-l", str(Path(a.bin_dir, build, "sd-cli").resolve()), "-m", a.model, "-p", PROMPT,
           "--steps", str(a.steps), "-W", str(a.res), "-H", str(a.res), "-s", "42", "--cfg-scale", "1", "-o", str(img),
           "-v", *STUDIO, "--conv-direct", mode]
    env = dict(os.environ, GGML_METAL_FORCE_SIMDGROUP="1", S03_CONV_LOG="1")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=timeout)
        log, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        log, rc = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace"), "timeout"
    wall = time.perf_counter() - t0
    (out / f"{tag}.log").write_text(" ".join(cmd) + "\n" + log)
    r = {"arm": arm, "tag": tag, "rc": rc, "wall_s": round(wall, 2),
         "decode_s": grab(r"decode_first_stage completed, taking ([0-9.]+)s", log),
         "sample_s": grab(r"sampling completed, taking ([0-9.]+)s", log),
         "conv_mm": log.count("S03_CONV2D mm"), "conv_scalar": log.count("S03_CONV2D scalar"),
         "rss_mib": (lambda v: v and round(v / 2**20, 1))(grab(r"(\d+)\s+maximum resident set size", log)),
         "img": img.name if img.exists() else None,
         "simdgroup_mm": "simdgroup matrix mul. = true" in log,
         "conv_auto": re.findall(r"conv2d direct auto: .*", log)}
    print(json.dumps(r), flush=True)
    return r


def psnr(a, b):
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    return (float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)), int(np.max(np.abs(x - y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="bin")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--slow-timeout", type=int, default=900)
    ap.add_argument("--no-slow", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    order = ["base_im2col", "head_im2col", "head_direct"]
    rows = [run("head_direct", a, out, "warm_head_direct", 1800)]  # warm-up (shader compile, page cache)
    for i in range(a.reps):
        seq = order if i % 2 == 0 else order[::-1]
        rows += [run(arm, a, out, f"{arm}_r{i + 1}", 1800) for arm in seq]
    if not a.no_slow:
        rows.append(run("base_direct", a, out, "base_direct_r1", a.slow_timeout))
    ref = next((out / r["img"] for r in rows if r["arm"] == "base_im2col" and r["img"]), None)
    for r in rows:
        if ref and r["img"]:
            r["psnr_vs_base_im2col"], r["maxabs"] = psnr(ref, out / r["img"])
            r["pix_sha"] = hashlib.sha256(Image.open(out / r["img"]).convert("RGB").tobytes()).hexdigest()[:16]
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    lines = [f"res={a.res} steps={a.steps} reps={a.reps}",
             "| arm | n ok | decode s med [min,max] | sample s med | conv mm/scalar per run | PSNR vs base_im2col (min) | maxabs | distinct px |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in ["base_im2col", "head_im2col", "head_direct", "base_direct"]:
        rs = [r for r in rows if r["arm"] == arm and not r["tag"].startswith("warm")]
        ok = [r for r in rs if r["rc"] == 0 and r["decode_s"] is not None]
        if not rs:
            continue
        d = [r["decode_s"] for r in ok]
        ps = [r.get("psnr_vs_base_im2col") for r in ok if r.get("psnr_vs_base_im2col") is not None]
        lines.append(f"| {arm} | {len(ok)}/{len(rs)} | " + (f"{st.median(d):.2f} [{min(d):.2f}, {max(d):.2f}]" if d else "-") +
                     f" | {st.median([r['sample_s'] for r in ok]) if ok else '-'} | "
                     f"{rs[0]['conv_mm']}/{rs[0]['conv_scalar']} | {min(ps) if ps else '-'} | "
                     f"{max((r.get('maxabs', 0) for r in ok), default='-')} | {len({r.get('pix_sha') for r in ok})} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
