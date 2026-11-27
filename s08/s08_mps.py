"""S08: MiniMax-H3 VAE encoder T=1 Conv3d -> Conv2d lowering on MPS.

Step 1 (op level): every unique encoder causal-conv geometry for a T=1 input (untiled 1024^2 and the released
256^2 encode tile), stock MiniMaxH3VideoCausalConv3d.forward vs Studio's _single_frame_conv_forward, fp32 + bf16,
ulp diffs vs each other and vs a CPU float64 oracle, median of N iters with torch.mps.synchronize.
Step 2: AutoencoderKLMiniMaxH3 with the real config (random weights; decoder_num_layers=1, the decoder is not
on the encode path) encoding a 1024^2 reference frame through the stock tiled encode, stock vs lowered.

usage: s08_mps.py --device mps --cfg vae_config.json --studio-dir DIR --out DIR [--res 1024] [--tile 256]
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import types

import torch
import torch.nn.functional as F


def load_studio(studio_dir):
    pkg = types.ModuleType("h3pkg")
    pkg.__path__ = [studio_dir]
    sys.modules["h3pkg"] = pkg
    for name in ("rocm_bf16", "video_minimax_h3_vae"):
        spec = importlib.util.spec_from_file_location(f"h3pkg.{name}", os.path.join(studio_dir, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"h3pkg.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["h3pkg.video_minimax_h3_vae"]


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def bf16_ordered(t):
    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -32768 - i, i)


def ulp_bf16_pair(a, b):
    """exact ulp distance between two bf16 tensors"""
    return (bf16_ordered(a) - bf16_ordered(b)).abs()


def ulp_bf16_of(ref):
    """bf16 ulp size at |ref| (ref float64); floor at the smallest normal"""
    m = ref.abs().clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(m)) - 7)


def err_in_bf16_ulps(x, ref, floor_frac=1e-3):
    """|x - ref| in bf16 ulps of ref; raw and with the ulp floored at floor_frac * max|ref| (cancellation points)"""
    ref = ref.double()
    d = (x.double() - ref).abs()
    u = ulp_bf16_of(ref)
    raw = (d / u)
    floored = d / torch.maximum(u, ulp_bf16_of(torch.tensor(floor_frac * ref.abs().max().item(), dtype=torch.float64)))
    return {
        "max_raw": raw.max().item(),
        "max_floored": floored.max().item(),
        "p999_raw": torch.quantile(raw.flatten()[:: max(1, raw.numel() // 2_000_000)].float(), 0.999).item(),
        "frac_gt1_raw": (raw > 1).double().mean().item(),
        "max_abs": d.max().item(),
        # elements off by more than 1e-3 of the output's range: a wrong result, not rounding
        "frac_bad": (d > 1e-3 * ref.abs().max()).double().mean().item(),
    }


def collect_geometries(H3, cfg, res):
    """causal-conv geometries the encoder sees for a (1,3,1,res,res) input, in call order, deduplicated"""
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoCausalConv3d, MiniMaxH3VideoEncoder3d

    with torch.device("meta"):
        enc = MiniMaxH3VideoEncoder3d(
            in_channels=cfg["in_channels"],
            out_channels=2 * cfg["latent_channels"],
            block_out_channels=tuple(cfg["block_out_channels"]),
            layers_per_block=cfg["layers_per_block"],
            spatial_downsample_factors=tuple(cfg["spatial_downsample_factors"]),
            temporal_downsample_factors=tuple(cfg["temporal_downsample_factors"]),
            norm_num_groups=cfg["norm_num_groups"],
            norm_eps=cfg["norm_eps"],
            spatial_padding_mode=cfg["spatial_padding_mode"],
        )
    seen, order = {}, []

    def hook(mod, args, name):
        x = args[0]
        key = (mod.in_channels, mod.out_channels, tuple(mod.kernel_size), tuple(mod.stride),
               int(mod.spatial_padding), int(mod.temporal_padding), tuple(x.shape))
        if key not in seen:
            seen[key] = [name]
            order.append(key)
        else:
            seen[key].append(name)

    for name, m in enc.named_modules():
        if isinstance(m, MiniMaxH3VideoCausalConv3d):
            m.register_forward_pre_hook(lambda mod, args, name=name: hook(mod, args, name))
    with torch.no_grad():
        enc(torch.empty(1, cfg["in_channels"], 1, res, res, device="meta"))
    return [(k, seen[k]) for k in order]


def time_fn(fn, dev, iters, warmup=3):
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def op_level(H3, cfg, dev, res, iters, label, rows):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoCausalConv3d

    geos = collect_geometries(H3, cfg, res)
    for key, names in geos:
        cin, cout, k, stride, sp, tp, shape = key
        if k[0] == 1:
            continue  # 1x1x1 shortcut: the lowering does not apply (kt == 1)
        torch.manual_seed(0)
        conv = MiniMaxH3VideoCausalConv3d(cin, cout, kernel_size=k, stride=stride, spatial_padding=sp,
                                          temporal_padding=tp, spatial_padding_mode=cfg["spatial_padding_mode"])
        x_cpu = torch.randn(shape)
        with torch.no_grad():
            # oracle: float64 CPU conv over the last tap (the other taps multiply exact zeros)
            w64, b64 = conv.weight.double(), conv.bias.double()
            xo = x_cpu.double()[:, :, 0]
            if sp:
                xo = F.pad(xo, (sp,) * 4, mode=cfg["spatial_padding_mode"])
            oracle = F.conv2d(xo, w64[:, :, -1], b64, stride=stride[1:]).unsqueeze(2)
            ref_cpu32 = conv(x_cpu)  # stock CPU fp32
        row = {"set": label, "cin": cin, "cout": cout, "k": k, "stride": stride, "in_shape": list(shape),
               "calls": len(names), "first": names[0]}
        row["cpu32_stock_vs_oracle_bf16ulps"] = err_in_bf16_ulps(ref_cpu32, oracle)
        for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            try:
                c = MiniMaxH3VideoCausalConv3d(cin, cout, kernel_size=k, stride=stride, spatial_padding=sp,
                                               temporal_padding=tp, spatial_padding_mode=cfg["spatial_padding_mode"])
                c.load_state_dict(conv.state_dict())
                c = c.to(dev, dt)
                x = x_cpu.to(dev, dt)
                stock = c.forward
                calls = {}
                lowered = H3._single_frame_conv_forward(c, calls)
                with torch.no_grad():
                    ys = stock(x)
                    yl = lowered(x)
                    sync(dev)
                    assert calls.get(H3.LEVER_ROCM_SINGLE_FRAME_ENCODER, 0) == 1, calls  # the lowered path ran
                    assert ys.shape == yl.shape, (ys.shape, yl.shape)
                    r = {"bitwise_equal": bool(torch.equal(ys, yl))}
                    if dt is torch.bfloat16:
                        u = ulp_bf16_pair(ys.cpu(), yl.cpu())
                        r["stock_vs_lowered_max_ulp"] = u.max().item()
                        r["stock_vs_lowered_frac_gt1ulp"] = (u > 1).double().mean().item()
                        r["stock_vs_lowered_frac_ne"] = (u > 0).double().mean().item()
                    else:
                        r["stock_vs_lowered_bf16ulps"] = err_in_bf16_ulps(yl.cpu(), ys.cpu().double())
                        r["stock_vs_lowered_max_abs"] = (ys - yl).abs().max().item()
                    r["stock_vs_oracle"] = err_in_bf16_ulps(ys.cpu(), oracle)
                    r["lowered_vs_oracle"] = err_in_bf16_ulps(yl.cpu(), oracle)
                    del ys, yl
                    t_s = time_fn(lambda: stock(x), dev, iters)
                    t_l = time_fn(lambda: lowered(x), dev, iters)
                    r["stock_ms"] = 1e3 * t_s[0]
                    r["lowered_ms"] = 1e3 * t_l[0]
                    r["stock_ms_minmax"] = [1e3 * t_s[1], 1e3 * t_s[2]]
                    r["lowered_ms_minmax"] = [1e3 * t_l[1], 1e3 * t_l[2]]
                    r["speedup"] = t_s[0] / t_l[0]
                del c, x
            except Exception as exc:  # noqa: BLE001
                r = {"error": f"{type(exc).__name__}: {exc}"[:400]}
            if dev == "mps":
                torch.mps.empty_cache()
            row[dt_name] = r
        rows.append(row)
        f = row.get("fp32", {})
        b = row.get("bf16", {})
        print(f"[{label}] {cin}->{cout} k{k} s{stride} in{tuple(shape)} x{len(names)}: "
              f"fp32 {f.get('stock_ms', float('nan')):.2f}->{f.get('lowered_ms', float('nan')):.2f}ms "
              f"({f.get('speedup', float('nan')):.2f}x) | bf16 {b.get('stock_ms', float('nan')):.2f}->"
              f"{b.get('lowered_ms', float('nan')):.2f}ms ({b.get('speedup', float('nan')):.2f}x) "
              f"maxulp {b.get('stock_vs_lowered_max_ulp')} err {f.get('error', '')}{b.get('error', '')}", flush=True)


def build_vae(cfg):
    from diffusers import AutoencoderKLMiniMaxH3

    kw = {k: v for k, v in cfg.items() if not k.startswith("_")}
    kw["decoder_num_layers"] = 1  # not on the encode path; keeps the random model small
    torch.manual_seed(0)
    vae = AutoencoderKLMiniMaxH3(**kw).eval()
    vae.decoder = torch.nn.Identity()  # encode only
    torch.manual_seed(1)
    img = torch.rand(1, 3, 1, cfg["_res"], cfg["_res"]) * 2 - 1
    return vae, img


def vae_arm(H3, cfg, dev, arm, dt_name, iters, out):
    """one arm (stock | lowered | cpuref) of a real-config random-weight encode; latent saved for compare"""
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoCausalConv3d

    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dt_name]
    vae, img = build_vae(cfg)
    r = {"arm": arm, "dtype": dt_name, "encoder_params_M": sum(p.numel() for p in vae.encoder.parameters()) / 1e6}
    if arm == "cpuref":
        dev, dt = "cpu", torch.float32
    v = vae.to(dev, dt)
    x = img.to(dev, dt)
    calls = {}
    if arm == "lowered":
        for m in v.encoder.modules():
            if isinstance(m, MiniMaxH3VideoCausalConv3d):
                m.forward = H3._single_frame_conv_forward(m, calls)
    t0 = time.perf_counter()
    with torch.no_grad():
        y = v.encode(x).latent_dist.mean
        sync(dev)
        r["first_s"] = time.perf_counter() - t0
        r["lowered_conv_calls_first_encode"] = calls.get(H3.LEVER_ROCM_SINGLE_FRAME_ENCODER, 0)
        if arm != "cpuref" and iters > 0:
            t = time_fn(lambda: v.encode(x), dev, iters, warmup=0)
            r["median_s"], r["min_s"], r["max_s"] = t
    if dev == "mps":
        r["mps_driver_alloc_GB"] = torch.mps.driver_allocated_memory() / 2**30
    r["latent_shape"] = list(y.shape)
    torch.save(y.cpu(), os.path.join(out, f"latent_{arm}_{dt_name}.pt"))
    json.dump(r, open(os.path.join(out, f"vae_{arm}_{dt_name}.json"), "w"), indent=1)
    print(f"[vae {arm} {dt_name}] {json.dumps(r)}", flush=True)


def vae_compare(out):
    ref = torch.load(os.path.join(out, "latent_cpuref_fp32.pt")).double()
    res = {}
    for dt_name in ("fp32", "bf16"):
        lat = {a: os.path.join(out, f"latent_{a}_{dt_name}.pt") for a in ("stock", "lowered")}
        if not all(os.path.exists(p) for p in lat.values()):
            continue
        ys, yl = (torch.load(lat[a]) for a in ("stock", "lowered"))
        r = {"bitwise_equal": bool(torch.equal(ys, yl)),
             "stock_vs_lowered_max_abs": (ys.double() - yl.double()).abs().max().item(),
             "stock_vs_cpu32": err_in_bf16_ulps(ys, ref), "lowered_vs_cpu32": err_in_bf16_ulps(yl, ref)}
        if dt_name == "fp32":
            r["stock_vs_lowered_bf16ulps"] = err_in_bf16_ulps(yl, ys.double())
        else:
            u = ulp_bf16_pair(ys.to(torch.bfloat16), yl.to(torch.bfloat16))
            r["stock_vs_lowered_max_ulp"], r["stock_vs_lowered_frac_gt1ulp"] = u.max().item(), (u > 1).double().mean().item()
        t = {a: json.load(open(os.path.join(out, f"vae_{a}_{dt_name}.json"))) for a in ("stock", "lowered")}
        r["stock_median_s"], r["lowered_median_s"] = t["stock"].get("median_s"), t["lowered"].get("median_s")
        if r["stock_median_s"] and r["lowered_median_s"]:
            r["speedup"] = r["stock_median_s"] / r["lowered_median_s"]
        r["lowered_conv_calls_first_encode"] = t["lowered"]["lowered_conv_calls_first_encode"]
        res[dt_name] = r
        print(f"[vae compare {dt_name}] {json.dumps(r)}", flush=True)
    json.dump(res, open(os.path.join(out, "vae_compare.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--studio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--vae-iters", type=int, default=5)
    ap.add_argument("--skip-full", action="store_true", help="skip the untiled res^2 op set")
    ap.add_argument("--skip-vae", action="store_true")
    ap.add_argument("--vae-arm", choices=["stock", "lowered", "cpuref", "compare"])
    ap.add_argument("--vae-dtype", default="fp32", choices=["fp32", "bf16"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    H3 = load_studio(a.studio_dir)
    cfg = json.load(open(a.cfg))
    import diffusers
    meta = {"torch": torch.__version__, "diffusers": diffusers.__version__, "device": a.device,
            "mps_available": torch.backends.mps.is_available(), "res": a.res, "tile": a.tile, "iters": a.iters}
    print(meta, flush=True)
    if a.device == "mps" and a.vae_arm not in ("cpuref", "compare") and not meta["mps_available"]:
        sys.exit("MPS not available: VOID")
    rows = []
    if a.vae_arm == "compare":
        vae_compare(a.out)
        return
    if a.vae_arm:
        cfg["_res"] = a.res
        vae_arm(H3, cfg, a.device, a.vae_arm, a.vae_dtype, a.vae_iters, a.out)
        return
    op_level(H3, cfg, a.device, a.tile, a.iters, f"tile{a.tile}", rows)
    json.dump({"meta": meta, "ops": rows}, open(os.path.join(a.out, "ops.json"), "w"), indent=1)
    if not a.skip_full:
        op_level(H3, cfg, a.device, a.res, a.iters, f"full{a.res}", rows)
        json.dump({"meta": meta, "ops": rows}, open(os.path.join(a.out, "ops.json"), "w"), indent=1)
    json.dump({"meta": meta, "ops": rows}, open(os.path.join(a.out, "ops.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
