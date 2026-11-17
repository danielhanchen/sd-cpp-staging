#!/usr/bin/env python3
"""armws remote driver (staging runners): A = Windows arm64 i8mm variants, B = work stealing in the static GEMMs.

Arms are branches of the public staging fork built with the fork's ggml-patches applied in glob order (as the
prebuilt workflow does) plus the qmm_bench mul_mat bench. Per leg:
  qmm    : qmm_bench (types x shapes, -r 3 inside) for every run label, 3 interleaved reps, order rotating
  render : 4-step FLUX.2 klein 4B Q4_K_M 512 px, same rotation; png compare vs the reference label
A run label may add a one-core hog (Linux: a busy loop pinned to the last CPU) or env (GGML_CPU_ARM_NO_I8MM=1).
Everything lands in ./art.
"""
import argparse, glob, json, os, platform, re, shlex, shutil, subprocess, sys, time
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
}
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
SPEED = ["--diffusion-fa", "--diffusion-conv-direct"]  # Studio native speed flags on CPU
IS_WIN = sys.platform == "win32"
EXE = ".exe" if IS_WIN else ""
NCPU = os.cpu_count() or 4
LOG = open(ART / "driver.log", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def sh(cmd, log=None, cwd=None, env=None, check=True):
    say("$", " ".join(map(str, cmd)))
    with open(log, "a", encoding="utf-8") if log else open(os.devnull, "w") as fh:
        p = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, stdout=fh if log else subprocess.PIPE,
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
    info = {"platform": platform.platform(), "machine": platform.machine(), "nproc": NCPU}
    if IS_WIN:
        info["cpu"] = sh(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"], check=False).strip()
        import ctypes
        k = ctypes.windll.kernel32
        info["pf"] = {n: bool(k.IsProcessorFeaturePresent(i)) for n, i in
                      (("dotprod", 43), ("sve", 46), ("i8mm", 66), ("fp16", 67), ("sme", 70))}
        info["winver"] = platform.version()
    else:
        info["cpu"] = sh(["sh", "-c", "lscpu | grep -E 'Model name|^CPU\\(s\\)|Flags' | cut -c1-600"], check=False)
    return info


KINDS = {
    # Windows arm64 bundle as master ships it: one static build (unsloth-sd-prebuilt.yml, win-cpu-arm64)
    "win_static": ["-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++", "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16",
                   "-DGGML_OPENMP=OFF", "-DSD_BUILD_SHARED_LIBS=ON", "-DGGML_NATIVE=OFF"],
    # Windows arm64 bundle as the head workflow builds it: CPU variants picked at startup
    "win_dl": ["-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++", "-DGGML_OPENMP=OFF", "-DSD_BUILD_SHARED_LIBS=ON",
               "-DSD_BUILD_SHARED_GGML_LIB=ON", "-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON",
               "-DGGML_NATIVE=OFF"],
    # Linux arm64 single-variant builds with the variant flags (armv8.2_2 / armv8.6_0)
    "arm82": ["-DCMAKE_C_COMPILER=gcc-14", "-DCMAKE_CXX_COMPILER=g++-14", "-DGGML_NATIVE=OFF",
              "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"],
    "arm86": ["-DCMAKE_C_COMPILER=gcc-14", "-DCMAKE_CXX_COMPILER=g++-14", "-DGGML_NATIVE=OFF",
              "-DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+fp16+i8mm"],
    "x64native": ["-DGGML_NATIVE=ON"],
    # Windows x64 bundle (MSVC): CPU variants picked at startup, as win-cpu-x64 builds it
    "win_x64_dl": ["-DCMAKE_CXX_FLAGS=/bigobj", "-DSD_BUILD_SHARED_LIBS=ON", "-DSD_BUILD_SHARED_GGML_LIB=ON", "-DBUILD_SHARED_LIBS=ON",
                   "-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON", "-DGGML_NATIVE=OFF"],
}


def build(arms, jobs):
    out = {}
    for spec in arms:  # name=branch:kind
        name, rest = spec.split("=", 1)
        branch, kind = rest.split(":")
        src = W / f"src_{name}"
        if src.exists():
            shutil.rmtree(src)
        sh(["git", "-c", "core.autocrlf=false", "clone", "-q", "--depth", "1", "-b", branch, REPO, str(src)])
        sh(["git", "-c", "core.autocrlf=false", "submodule", "update", "--init", "--depth", "1", "ggml"], cwd=src)
        for p in sorted(glob.glob(str(src / "scripts/unsloth/ggml-patches/*.patch"))):
            sh(["git", "apply", os.path.relpath(p, src / "ggml")], cwd=src / "ggml")
        rev = sh(["git", "rev-parse", "HEAD"], cwd=src).strip()
        shutil.copy(HERE / "qmm_bench.cpp", src / "ggml/tests/qmm_bench.cpp")
        with open(src / "ggml/tests/CMakeLists.txt", "a") as f:
            f.write("\nadd_executable(qmm_bench qmm_bench.cpp)\ntarget_link_libraries(qmm_bench PRIVATE ggml)\n")
        b = W / f"build_{name}"
        log = ART / f"build_{name}.log"
        t0 = time.time()
        sh(["cmake", "-S", str(src), "-B", str(b), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DSD_BUILD_EXAMPLES=ON",
            "-DSD_SERVER_BUILD_FRONTEND=OFF", "-DSD_WEBP=OFF", "-DSD_WEBM=OFF", "-DGGML_BUILD_TESTS=ON", *KINDS[kind]], log=log)
        sh(["cmake", "--build", str(b), "-j", str(jobs), "--target", "sd-cli", "qmm_bench"], log=log)
        keep = W / f"bin_{name}"
        if keep.exists():
            shutil.rmtree(keep)
        shutil.copytree(b / "bin", keep)
        out[name] = {"branch": branch, "kind": kind, "rev": rev, "build_s": round(time.time() - t0, 1), "bin": str(keep),
                     "files": sorted(os.listdir(keep))}
        say("built", name, json.dumps(out[name]))
    return out


def parse_runs(spec):
    # label=arm[+hog][+noi8mm]
    runs = []
    for item in spec.split(","):
        label, rest = item.split("=", 1)
        arm, *mods = rest.split("+")
        env = {"GGML_CPU_IQK_DEBUG": "1"}
        if "noi8mm" in mods:
            env["GGML_CPU_ARM_NO_I8MM"] = "1"
        runs.append({"label": label, "arm": arm, "hog": "hog" in mods, "env": env})
    return runs


class Hog:
    def __init__(self, on):
        self.on, self.p = on and not IS_WIN, None

    def __enter__(self):
        if self.on:
            self.p = subprocess.Popen(["taskset", "-c", str(NCPU - 1), "sh", "-c", "while :; do :; done"])
            time.sleep(0.5)
        return self

    def __exit__(self, *a):
        if self.p:
            self.p.kill(); self.p.wait()


def run_env(r):
    return dict(os.environ, **r["env"])


def qmm(bins, runs, reps, types, shapes):
    out = ART / "qmm"; out.mkdir(exist_ok=True)
    rows = []
    for rep in range(reps):
        k = rep % len(runs)
        for r in runs[k:] + runs[:k]:
            exe = str(Path(bins[r["arm"]]["bin"]) / f"qmm_bench{EXE}")
            cmd = [exe, "-t", str(NCPU), "-r", "3", "-T", types, "-b", "both", "--oracle-cols", "16"]
            for s in shapes:
                cmd += ["-s", s]
            with Hog(r["hog"]):
                t0 = time.time()
                p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=run_env(r))
                wall = time.time() - t0
            (out / f"{r['label']}_r{rep}.log").write_text(p.stdout + "\n--- stderr\n" + p.stderr, encoding="utf-8")
            n = 0
            for line in p.stdout.splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                d.update(label=r["label"], rep=rep); rows.append(d); n += 1
            say("qmm", r["label"], "rep", rep, "rc", p.returncode, "rows", n, f"wall {wall:.0f}s")
        json.dump(rows, open(ART / "qmm.json", "w"), indent=1)


PAT = {"te_s": r"get_learned_condition completed, taking ([\d.]+)s", "sampling_s": r"sampling completed, taking ([\d.]+)s",
       "decode_s": r"decode_first_stage completed, taking ([\d.]+)s", "generate_s": r"generate_(?:image|video) completed in ([\d.]+)s"}


def render_one(bins, r, pre):
    args = ["--diffusion-model", fetch("k2"), "--llm", fetch("q3"), "--vae", fetch("v2"), "--cfg-scale", "1", "--steps", "4",
            "-W", "512", "-H", "512"]
    exe = str(Path(bins[r["arm"]]["bin"]) / f"sd-cli{EXE}")
    cmd = [exe, *args, "-p", PROMPT, "--seed", "42", "-o", f"{pre}.png", "-v", *SPEED, "-t", str(NCPU)]
    with Hog(r["hog"]):
        t0 = time.time()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=run_env(r))
        wall = time.time() - t0
    Path(f"{pre}.log").write_text(" ".join(shlex.quote(x) for x in cmd) + "\n" + p.stdout, encoding="utf-8")
    d = {k: float(m[-1]) for k, pat in PAT.items() if (m := re.findall(pat, p.stdout))}
    m = re.findall(r"loaded CPU backend from (\S+)", p.stdout)
    d.update(rc=p.returncode, wall=round(wall, 2), cpu_backend=m[-1] if m else "static")
    return d


def compare(a, b):
    if not (os.path.exists(a) and os.path.exists(b)):
        return {"error": "missing"}
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {"bytes_equal": open(a, "rb").read() == open(b, "rb").read()}
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64); y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return {"bytes_equal": open(a, "rb").read() == open(b, "rb").read(), "pixels_equal": bool((x == y).all()),
            "max_abs_px": float(abs(x - y).max()), "psnr_db": (10 * np.log10(255 ** 2 / mse)) if mse else float("inf")}


def render(bins, runs, reps, ref):
    out = ART / "render"; out.mkdir(exist_ok=True)
    rows = []
    for rep in range(reps):
        k = rep % len(runs)
        for r in runs[k:] + runs[:k]:
            d = render_one(bins, r, str(out / f"{r['label']}_r{rep}"))
            d.update(label=r["label"], rep=rep); rows.append(d); say("render", json.dumps(d))
            json.dump(rows, open(ART / "render.json", "w"), indent=1)
    cmp = {}
    for r in runs:
        cmp[r["label"]] = {f"r{rep}": compare(str(out / f"{ref}_r0.png"), str(out / f"{r['label']}_r{rep}.png")) for rep in range(reps)}
    say("cmp vs", ref, json.dumps(cmp))
    json.dump(cmp, open(ART / "render_cmp.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--types", default="q8_0,q4_0,q5_0,q4_K,q5_K,q6_K,q3_K,iq4_xs")
    ap.add_argument("--shapes", default="3072,3072,1024/3072,12288,1024")
    ap.add_argument("--steps", default="qmm,render")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    info = host_info(); json.dump(info, open(ART / "host.json", "w"), indent=1); say(json.dumps(info)[:900])
    bins = build(a.arms.split(","), NCPU)
    json.dump(bins, open(ART / "bins.json", "w"), indent=1)
    runs = parse_runs(a.runs)
    steps = a.steps.split(",")
    if "render" in steps:
        for k in FILES:
            fetch(k)
    for step, fn in (("qmm", lambda: qmm(bins, runs, a.reps, a.types, a.shapes.split("/"))),
                     ("render", lambda: render(bins, runs, a.reps, a.ref))):
        if step in steps:
            try:
                t0 = time.time(); fn(); say(f"step {step} done in {time.time() - t0:.0f}s")
            except Exception as e:  # noqa: BLE001
                say(f"step {step} FAILED: {e}"); (ART / f"{step}_FAILED.txt").write_text(str(e))
    say("=== DONE ===")


if __name__ == "__main__":
    main()
