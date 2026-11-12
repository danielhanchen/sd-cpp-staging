#!/usr/bin/env python3
"""cpu3 remote driver (staging GitHub runners: macos-15 M1, ubuntu x64, ubuntu arm64).

Arms are built from branches of the public staging fork with the fork's ggml-patches applied in glob order
(as unsloth-sd-prebuilt.yml does), plus optional extra patches from this directory (the scratch per-node
profiler for the `prof` arm). Steps:
  prof : per-node CPU profile (GGML_CPU_PROF=1) of the cases on the prof arm
  ab   : base vs head(s) interleaved, order rotating, with a second base arm (A/A); pixel compare vs base
Everything lands in ./art.
"""
import argparse, glob, json, os, platform, re, shlex, shutil, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path.cwd()
ART = ROOT / "art"; ART.mkdir(exist_ok=True)
W = ROOT / "w"; W.mkdir(exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
HERE = Path(__file__).resolve().parent
REPO = "https://github.com/danielhanchen/sd-cpp-staging"
HF = "https://huggingface.co"
FILES = {
    "k2": "unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf",
    "q3": "unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
    "v2": "unsloth/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors",
    "k3": "unsloth/flux-3-image-klein-GGUF/resolve/main/flux-3-image-klein-Q4_K_M.gguf",
    "qvl": "Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
    "wan": "samuelchristlie/Wan2.1-T2V-1.3B-GGUF/resolve/main/Wan2.1-T2V-1.3B-Q4_K_M.gguf",
    "wvae": "Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
    "umt5": "city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q4_K_M.gguf",
}
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
# Wan's default negative prompt (docs/wan.md), the long one Studio users get
WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
           "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
SPEED = ["--diffusion-fa", "--diffusion-conv-direct"]  # Studio native speed flags on CPU
LOG = open(ART / "driver.log", "a")
IS_MAC = sys.platform == "darwin"


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def sh(cmd, log=None, cwd=None, env=None, check=True):
    say("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))
    with open(log, "a") if log else open(os.devnull, "w") as fh:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env, stdout=fh if log else subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, errors="replace")
    if check and p.returncode:
        tail = Path(log).read_text(errors="replace")[-4000:] if log else (p.stdout or "")[-4000:]
        raise RuntimeError(f"rc={p.returncode}: {cmd}\n{tail}")
    return p.stdout if not log else ""


def fetch(key):
    name = FILES[key].split("/")[-1]
    dst = MODELS / f"{key}__{name}"
    if not dst.exists():
        sh(["curl", "-fsSL", "--retry", "4", "-o", str(dst) + ".part", f"{HF}/{FILES[key]}"])
        os.replace(str(dst) + ".part", dst)
    return str(dst)


def host_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "nproc": os.cpu_count()}
    if IS_MAC:
        info["cpu"] = sh("sysctl -n machdep.cpu.brand_string", check=False).strip()
        info["mem"] = sh("sysctl -n hw.memsize", check=False).strip()
    else:
        info["cpu"] = sh("lscpu | grep -E 'Model name|^CPU\\(s\\)|Flags' | cut -c1-400", check=False)
    return info


def build(defines, jobs, arms):
    env = dict(os.environ, CCACHE_BASEDIR=str(W), CCACHE_NOHASHDIR="true", CCACHE_DIR=str(ROOT / "ccache"))
    launcher = ["-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"] if shutil.which("ccache") else []
    out = {}
    for spec in arms:  # name=branch[+patch[+patch]]
        name, rest = spec.split("=", 1)
        branch, *extra = rest.split("+")
        src = W / "src"
        if src.exists():
            shutil.rmtree(src)
        sh(["git", "clone", "-q", "--depth", "1", "-b", branch, REPO, str(src)])
        sh(["git", "submodule", "update", "--init", "--depth", "1", "ggml"], cwd=src)
        for p in sorted(glob.glob(str(src / "scripts/unsloth/ggml-patches/*.patch"))):
            sh(["git", "-C", "ggml", "apply", os.path.relpath(p, src / "ggml")], cwd=src)
        for e in extra:  # prof_ggml.patch applies inside ggml/, prof_sd.patch at the root
            sh(["git", "apply", str(HERE / e)], cwd=src / "ggml" if "ggml" in e else src)
        rev = sh(["git", "rev-parse", "HEAD"], cwd=src).strip()
        b = W / "build"
        log = ART / f"build_{name}.log"
        t0 = time.time()
        sh(["cmake", "-S", str(src), "-B", str(b), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DSD_BUILD_EXAMPLES=ON",
            "-DSD_SERVER_BUILD_FRONTEND=OFF", "-DSD_WEBP=OFF", "-DSD_WEBM=OFF", *defines, *launcher], log=log, env=env)
        sh(["cmake", "--build", str(b), "-j", str(jobs), "--target", "sd-cli"], log=log, env=env)
        keep = W / f"bin_{name}"
        if keep.exists():
            shutil.rmtree(keep)
        shutil.copytree(b / "bin", keep)
        out[name] = {"branch": branch, "extra": extra, "rev": rev, "build_s": round(time.time() - t0, 1), "bin": str(keep)}
        say("built", name, out[name])
    return out


def case_args(c):
    k2, q3, v2 = fetch("k2"), fetch("q3"), fetch("v2")
    if c.startswith("k2_"):
        s, steps = c.split("_")[1], {"512": 2, "1024": 1}[c.split("_")[1]]
        return ["--diffusion-model", k2, "--llm", q3, "--vae", v2, "--cfg-scale", "1", "--steps", str(steps), "-W", s, "-H", s], steps, "png"
    if c.startswith("k3_"):
        s, steps = c.split("_")[1], {"512": 2, "1024": 1}[c.split("_")[1]]
        return ["--diffusion-model", fetch("k3"), "--llm", fetch("qvl"), "--vae", v2, "--cfg-scale", "1", "--steps", str(steps),
                "-W", s, "-H", s, "--sampling-method", "euler"], steps, "png"
    if c == "wan":
        return ["-M", "vid_gen", "--diffusion-model", fetch("wan"), "--t5xxl", fetch("umt5"), "--vae", fetch("wvae"), "--cfg-scale", "6",
                "--steps", "2", "-W", "480", "-H", "320", "--video-frames", "9", "--flow-shift", "3.0", "--sampling-method", "euler",
                "-n", "blurry"], 2, "%03d.png"
    if c == "wan_small":
        return ["-M", "vid_gen", "--diffusion-model", fetch("wan"), "--t5xxl", fetch("umt5"), "--vae", fetch("wvae"), "--cfg-scale", "6",
                "--steps", "2", "-W", "320", "-H", "192", "--video-frames", "5", "--flow-shift", "3.0", "--sampling-method", "euler",
                "-n", WAN_NEG], 2, "%03d.png"
    raise ValueError(c)


PAT = {"te_s": r"get_learned_condition completed, taking ([\d.]+)s", "sampling_s": r"sampling completed, taking ([\d.]+)s",
       "decode_s": r"decode_first_stage completed, taking ([\d.]+)s", "generate_s": r"generate_(?:image|video) completed in ([\d.]+)s"}


def run_case(exe, c, out_prefix, threads, extra_flags, env=None):
    args, steps, ext = case_args(c)
    cmd = [exe, *args, "-p", PROMPT, "--seed", "42", "-o", f"{out_prefix}.{ext}", "-v", *SPEED, *extra_flags]
    if threads:
        cmd += ["-t", str(threads)]
    timev = (["/usr/bin/time", "-l"] if IS_MAC else ["/usr/bin/time", "-v"]) if os.path.exists("/usr/bin/time") else []
    t0 = time.time()
    p = subprocess.run(timev + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=env or os.environ)
    wall = time.time() - t0
    Path(f"{out_prefix}.log").write_text(" ".join(shlex.quote(x) for x in cmd) + "\n" + p.stdout)
    r = {k: float(m[-1]) for k, pat in PAT.items() if (m := re.findall(pat, p.stdout))}
    if "sampling_s" in r:
        r["s_per_step"] = r["sampling_s"] / steps
    r.update(rc=p.returncode, wall=wall)
    return r


def prof(bins, cases, threads, flags):
    out = ART / "prof"; out.mkdir(exist_ok=True)
    exe = str(Path(bins["prof"]["bin"]) / "sd-cli")
    for c in cases:
        r = run_case(exe, c, str(out / c), threads, flags, env=dict(os.environ, GGML_CPU_PROF="1"))
        say("prof", c, json.dumps(r))


def compare(a_prefix, b_prefix):
    import numpy as np
    from PIL import Image
    fa = sorted(glob.glob(a_prefix + "*.png")); fb = sorted(glob.glob(b_prefix + "*.png"))
    if not fa or len(fa) != len(fb):
        return {"error": f"frames {len(fa)} vs {len(fb)}"}
    res = {"frames": len(fa), "bytes_equal": True, "pixels_equal": True, "max_abs_px": 0.0, "psnr_db_min": float("inf")}
    for x, y in zip(fa, fb):
        res["bytes_equal"] &= open(x, "rb").read() == open(y, "rb").read()
        a = np.asarray(Image.open(x).convert("RGB"), dtype=np.float64); b = np.asarray(Image.open(y).convert("RGB"), dtype=np.float64)
        mse = float(((a - b) ** 2).mean())
        res["pixels_equal"] &= bool((a == b).all()); res["max_abs_px"] = max(res["max_abs_px"], float(abs(a - b).max()))
        res["psnr_db_min"] = min(res["psnr_db_min"], (10 * np.log10(255 ** 2 / mse)) if mse else float("inf"))
    return res


def ab(bins, cases, threads, reps, flags, arms):
    out = ART / "ab"; out.mkdir(exist_ok=True)
    rows = []
    labels = ["base", "base_aa"] + [a for a in arms if a not in ("base", "prof")]
    for c in cases:
        for rep in range(reps):
            k = rep % len(labels)
            for lab in labels[k:] + labels[:k]:
                arm = "base" if lab == "base_aa" else lab
                pre = str(out / f"{c}_{lab}_r{rep}")
                r = run_case(str(Path(bins[arm]["bin"]) / "sd-cli"), c, pre, threads, flags)
                r.update(case=c, arm=lab, rep=rep); say("ab", json.dumps(r)); rows.append(r)
        num = {}
        for lab in labels[1:]:
            try:
                num[lab] = compare(str(out / f"{c}_base_r0"), str(out / f"{c}_{lab}_r0"))
            except Exception as e:  # noqa: BLE001
                num[lab] = {"error": str(e)[:300]}
            say("cmp", c, "base vs", lab, json.dumps(num[lab]))
        rows.append({"case": c, "numerics": num})
        json.dump(rows, open(ART / "ab.json", "w"), indent=1)
        for f in out.glob(f"{c}_*_r[1-9]*.png"):  # keep r0 images only
            f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defines", default="")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--arms", default="base=cpu3-base,prof=cpu3-base+prof_ggml.patch+prof_sd.patch")
    ap.add_argument("--steps", default="build,prof")
    ap.add_argument("--flags", default="")
    ap.add_argument("--prof-cases", default="k2_512,k2_1024,k3_512,wan")
    ap.add_argument("--ab-cases", default="k2_512")
    ap.add_argument("--ab-reps", type=int, default=3)
    a = ap.parse_args()
    steps = a.steps.split(",")
    info = host_info(); json.dump(info, open(ART / "host.json", "w"), indent=1); say(json.dumps(info)[:600])
    bins = build(shlex.split(a.defines), a.jobs, a.arms.split(","))
    json.dump(bins, open(ART / "bins.json", "w"), indent=1)
    flags = shlex.split(a.flags)
    for step, fn in (("prof", lambda: prof(bins, a.prof_cases.split(","), a.threads, flags)),
                     ("ab", lambda: ab(bins, a.ab_cases.split(","), a.threads, a.ab_reps, flags, list(bins)))):
        if step in steps:
            try:
                t0 = time.time(); fn(); say(f"step {step} done in {time.time() - t0:.0f}s")
            except Exception as e:  # noqa: BLE001
                say(f"step {step} FAILED: {e}"); (ART / f"{step}_FAILED.txt").write_text(str(e))
    say("=== DONE ===")


if __name__ == "__main__":
    main()
