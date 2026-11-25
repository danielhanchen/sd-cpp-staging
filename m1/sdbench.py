#!/usr/bin/env python3
"""sd.cpp #2403 A/B: one sd-cli binary, arms none (no LoRA) / runtime (--lora-apply-mode at_runtime) /
immediate (--lora-apply-mode immediately), interleaved rounds, fixed seed. Parses sampling time and s/it,
PSNR immediate vs runtime per round."""
import argparse, json, math, os, re, subprocess, sys, time
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model-args", required=True, help="model flags (string)")
ap.add_argument("--lora-dir", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--extra", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
P = "a red fox sitting in fresh snow, golden hour, detailed photograph"
ARMS = {"none": ("", None), "runtime": (f"<lora:{a.lora}:1>", "at_runtime"), "immediate": (f"<lora:{a.lora}:1>", "immediately")}
ORD = [["none", "runtime", "immediate"], ["immediate", "none", "runtime"], ["runtime", "immediate", "none"]]
rows = []
for r in range(a.rounds):
    for arm in ORD[r % 3]:
        tag, mode = ARMS[arm]
        png = f"{a.out}/{arm}_r{r}.png"
        cmd = [a.bin] + a.model_args.split() + ["-p", P + tag, "-W", str(a.size), "-H", str(a.size), "--steps", str(a.steps),
               "--cfg-scale", "1.0", "--seed", "42", "-o", png, "-v"] + a.extra.split()
        if mode:
            cmd += ["--lora-model-dir", a.lora_dir, "--lora-apply-mode", mode]
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t
        log = p.stdout + p.stderr
        open(f"{a.out}/{arm}_r{r}.log", "w").write(" ".join(cmd) + "\n" + log)
        samp = [float(x) for x in re.findall(r"sampling completed, taking ([0-9.]+)s", log)]
        sit = re.findall(r"([0-9.]+)(s/it|it/s)", log)
        lora_lines = [l.strip()[:200] for l in log.splitlines() if re.search(r"lora|LoRA|merge", l)][:12]
        row = {"arm": arm, "round": r, "rc": p.returncode, "wall_s": round(wall, 2), "sampling_s": samp,
               "last_rate": sit[-1] if sit else None, "lora_log": lora_lines,
               "gen_s": [float(x) for x in re.findall(r"generate_image completed in ([0-9.]+)s", log)]}
        print("ROW", json.dumps(row), flush=True)
        if p.returncode != 0:
            print(log[-3000:], flush=True)
        rows.append(row)


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) if os.path.exists(p) else None


def psnr(x, y):
    if x is None or y is None:
        return None
    m = ((x - y) ** 2).mean()
    return float("inf") if m == 0 else 10 * math.log10(255 ** 2 / m)


ps = {}
for r in range(a.rounds):
    ps[r] = {"imm_vs_runtime": psnr(load(f"{a.out}/immediate_r{r}.png"), load(f"{a.out}/runtime_r{r}.png")),
             "runtime_vs_none": psnr(load(f"{a.out}/runtime_r{r}.png"), load(f"{a.out}/none_r{r}.png")),
             "runtime_r_vs_r0": psnr(load(f"{a.out}/runtime_r{r}.png"), load(f"{a.out}/runtime_r0.png"))}
json.dump({"rows": rows, "psnr": ps}, open(f"{a.out}/summary.json", "w"), indent=1)
print("PSNR", json.dumps(ps))
