#!/usr/bin/env python3
"""One fresh process = one arm. Krea-2's VAE (Qwen/Qwen-Image vae, AutoencoderKLQwenImage) on MPS: N untiled decodes
(stock 5-D F.pad in every QwenImageCausalConv3d) or with Studio's zero-buffer causal conv
(diffusion_device._zero_buffer_causal_conv, origin/main verbatim, class added to _CAUSAL_PAD_CLASSES). Records
torch.mps.driver_allocated_memory after each decode and the output against a CPU float32 decode."""
import argparse
import json
import sys
import time
import types

ap = argparse.ArgumentParser()
ap.add_argument("--dd", required = True)
ap.add_argument("--arm", choices = ("stock", "zero_buffer"), required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--latent", type = int, default = 64)
ap.add_argument("--n", type = int, default = 20)
ap.add_argument("--frames", type = int, default = 1)
a = ap.parse_args()
sys.path.insert(0, a.dd)
import torch  # noqa: E402
from diffusers import AutoencoderKLQwenImage  # noqa: E402

import diffusion_device as D  # noqa: E402

vae = AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder = "vae", torch_dtype = torch.float32).eval()
g = torch.Generator().manual_seed(0)
z = torch.randn(1, 16, a.frames, a.latent, a.latent, generator = g)
with torch.no_grad():
    ref = vae.decode(z).sample
vae = vae.to("mps")
patched = 0
if a.arm == "zero_buffer":
    D._CAUSAL_PAD_CLASSES = frozenset(set(D._CAUSAL_PAD_CLASSES) | {"QwenImageCausalConv3d"})
    pipe = types.SimpleNamespace(components = {"vae": vae})
    target = types.SimpleNamespace(device = "mps")
    patched = D.install_frame_pad_fix(pipe, target)
rec = {"arm": a.arm, "torch": torch.__version__, "patched": patched, "driver_mib": [], "ms": []}
zm = z.to("mps")
with torch.no_grad():
    for i in range(a.n):
        t = time.perf_counter()
        y = vae.decode(zm).sample
        torch.mps.synchronize()
        rec["ms"].append(round(1000 * (time.perf_counter() - t), 1))
        rec["driver_mib"].append(round(torch.mps.driver_allocated_memory() / 2**20, 1))
yc = y.float().cpu()
d = (yc - ref).double()
rec["corr_vs_cpu"] = float(torch.corrcoef(torch.stack([yc.flatten().double(), ref.flatten().double()]))[0, 1])
rec["max_abs_vs_cpu"] = float(d.abs().max())
rec["out_sha"] = __import__("hashlib").sha256(yc.numpy().tobytes()).hexdigest()[:16]
json.dump(rec, open(a.out, "w"), indent = 1)
print(json.dumps({k: rec[k] for k in ("arm", "torch", "patched", "corr_vs_cpu", "max_abs_vs_cpu", "out_sha")}),
      "driver first/last", rec["driver_mib"][0], rec["driver_mib"][-1], "ms med", sorted(rec["ms"])[len(rec["ms"]) // 2])
