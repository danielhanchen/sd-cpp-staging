#!/usr/bin/env python3
"""One fresh process = one arm run: Studio's MPS allocator setup (diffusion_device._mps_or_cpu_target, verbatim from
unsloth origin/main) + PYTORCH_MPS_LOW_WATERMARK_RATIO from the environment (unset = torch default 1.4), then a diffusers
render. Samples torch.mps.driver_allocated_memory / current_allocated_memory every 20 ms."""
import argparse, json, os, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--dd", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--repo", default="stabilityai/sd-turbo")
ap.add_argument("--size", type=int, default=768)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--probe", action="store_true", help="print Studio's cap ratio and exit")
a = ap.parse_args()
sys.path.insert(0, a.dd)
import torch  # noqa: E402
import diffusion_device as D  # noqa: E402

tgt = D._mps_or_cpu_target(torch)
rec_b = int(torch.mps.recommended_max_memory())
ratio = D.mps_high_watermark_ratio(rec_b, D._total_ram_bytes())
if a.probe:
    print(json.dumps({"ratio": ratio, "recommended": rec_b, "total": D._total_ram_bytes(), "device": tgt.device}))
    sys.exit(0)
R = {"arm": a.arm, "low_env": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
     "high_env": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"), "cap_ratio": ratio, "recommended_mib": rec_b / 2**20,
     "total_mib": (D._total_ram_bytes() or 0) / 2**20, "device": tgt.device, "dtype": str(tgt.dtype), "torch": torch.__version__}
PK = {"phase": "load", "load": [0, 0], "render": [0, 0]}
stop = threading.Event()


def samp():
    while not stop.is_set():
        try:
            d, c = torch.mps.driver_allocated_memory(), torch.mps.current_allocated_memory()
            p = PK[PK["phase"]]
            p[0] = max(p[0], d); p[1] = max(p[1], c)
        except Exception:
            pass
        time.sleep(0.02)


threading.Thread(target=samp, daemon=True).start()
from diffusers import AutoPipelineForText2Image  # noqa: E402

t = time.time()
pipe = AutoPipelineForText2Image.from_pretrained(a.repo, torch_dtype=tgt.dtype, variant="fp16").to("mps")
pipe.set_progress_bar_config(disable=True)
torch.mps.synchronize(); R["load_s"] = time.time() - t
PK["phase"] = "render"
kw = dict(prompt="a red fox sitting in fresh snow, golden hour, detailed photograph", width=a.size, height=a.size,
          num_inference_steps=a.steps, guidance_scale=0.0)
img = pipe(generator=torch.Generator("cpu").manual_seed(1), **kw).images[0]  # warmup
torch.mps.synchronize()
ts = []
for i in range(a.n):
    t = time.time()
    img = pipe(generator=torch.Generator("cpu").manual_seed(1), **kw).images[0]
    torch.mps.synchronize(); ts.append(time.time() - t)
time.sleep(0.1); stop.set()
import numpy as np  # noqa: E402
arr = np.asarray(img, dtype=np.float32)
R.update(s_per_image=ts, driver_peak_mib_load=PK["load"][0] / 2**20, cur_peak_mib_load=PK["load"][1] / 2**20,
         driver_peak_mib_render=PK["render"][0] / 2**20, cur_peak_mib_render=PK["render"][1] / 2**20,
         driver_end_mib=torch.mps.driver_allocated_memory() / 2**20, img_mean=float(arr.mean()), img_std=float(arr.std()))
img.save(os.path.splitext(a.out)[0] + ".png")
json.dump(R, open(a.out, "w"), indent=1)
print("RESULT", json.dumps(R))
