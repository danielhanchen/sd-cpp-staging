"""C17-3 / C17-4 driver for one ubuntu-24.04-arm job (one model).

Builds unslothai/stable-diffusion.cpp at SDREF with the ggml-patches applied and the Linux aarch64 prebuilt
leg's cmake flags (all CPU variants, GGML_BACKEND_DL), twice: base, and base + llama.cpp #32554 (NEON q8_K
4x4/4x8 activation quant) on the vendored ggml. Then:
  1. one profiled base render: perf attached for the sampling phase only (between "generating image" and
     "sampling completed"), cpu-clock + dwarf call graphs; FA and q8_K inclusive / self shares from perf script.
  2. if the q8_K share >= GATE (or --force-ab): ABBA x rounds unprofiled renders, sha256 of PNG file + pixels.
Everything lands in ./art.
"""
import argparse, collections, hashlib, json, os, re, shutil, signal, subprocess, sys, threading, time, zlib, struct
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = Path.cwd(); W = ROOT / "w"; ART = ROOT / "art"
W.mkdir(exist_ok=True); ART.mkdir(exist_ok=True)
FORK = "https://github.com/unslothai/stable-diffusion.cpp"
HF = "https://huggingface.co"
TE = f"{HF}/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
MODELS = {
    "klein": {"dm": f"{HF}/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf",
              "vae": f"{HF}/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors"},
    "zimage": {"dm": f"{HF}/unsloth/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-Q4_K_M.gguf",
               "vae": f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"},
}
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
LOG = open(ART / "driver.log", "a")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def sh(cmd, log=None, cwd=None, check=True):
    say("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))
    with open(log or os.devnull, "a") as fh:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, stdout=fh if log else subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, errors="replace")
    if check and p.returncode:
        tail = Path(log).read_text(errors="replace")[-4000:] if log else (p.stdout or "")[-4000:]
        raise RuntimeError(f"rc={p.returncode}: {cmd}\n{tail}")
    return p.stdout or ""


def make_src(ref):
    base = W / "src_base"
    sh(["git", "clone", "-q", FORK, str(base)])
    sh(["git", "-C", str(base), "checkout", "-q", ref])
    sh(["git", "-C", str(base), "submodule", "update", "--init", "--depth", "1", "ggml"])
    patches = sorted((base / "scripts/unsloth/ggml-patches").glob("*.patch"))
    for p in patches:
        sh(["git", "-C", str(base / "ggml"), "apply", f"../scripts/unsloth/ggml-patches/{p.name}"], log=str(ART / "patch.log"))
    pat = W / "src_pat"
    shutil.copytree(base, pat, symlinks=True)
    sh(["git", "-C", str(pat / "ggml"), "apply", "-p2", "--verbose", str(H / "pr32554.diff")], log=str(ART / "patch_pr.log"))
    meta = {"head": sh(["git", "-C", str(base), "rev-parse", "HEAD"]).strip(),
            "ggml": sh(["git", "-C", str(base / "ggml"), "rev-parse", "HEAD"]).strip(), "patches": [p.name for p in patches],
            "pr_diff_sha256": hashlib.sha256((H / "pr32554.diff").read_bytes()).hexdigest()}
    (ART / "src.json").write_text(json.dumps(meta, indent=1)); say("src", meta["head"], meta["ggml"], len(patches))
    return {"A": base, "B": pat}


def build(name, src):
    bd = src / "build"; log = str(ART / f"build_{name}.log")
    defs = ["-DCMAKE_BUILD_TYPE=Release", "-DSD_BUILD_EXAMPLES=ON", "-DSD_SERVER_BUILD_FRONTEND=OFF", "-DSD_WEBP=OFF",
            "-DSD_WEBM=OFF", "-DGGML_NATIVE=OFF", "-DGIT_EXE=",
            "-DSD_BUILD_SHARED_LIBS=ON", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            "-DSD_BUILD_SHARED_GGML_LIB=ON", "-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON",
            "-DCMAKE_C_COMPILER=gcc-14", "-DCMAKE_CXX_COMPILER=g++-14"]
    if shutil.which("ccache"):
        defs += ["-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]
    t = time.time()
    sh(["cmake", "-S", str(src), "-B", str(bd), "-G", "Ninja"] + defs, log=log)
    sh(["cmake", "--build", str(bd), "-j", str(os.cpu_count()), "--target", "sd-cli"], log=log)
    say(f"built {name} in {time.time() - t:.0f}s")
    return bd / "bin"


def fetch_models(model):
    d = W / "models"; d.mkdir(exist_ok=True); out = {}
    for k, u in (("dm", MODELS[model]["dm"]), ("vae", MODELS[model]["vae"]), ("te", TE)):
        f = d / f"{model}_{k}{Path(u).suffix}"
        for _ in range(5):
            if f.exists() and f.stat().st_size > 1e6:
                break
            subprocess.run(["curl", "-sSfL", "--retry", "5", "-o", str(f), u]); time.sleep(3)
        out[k] = str(f); say("model", k, f.stat().st_size if f.exists() else None, u)
    return out


def png_pixels(path):
    data = Path(path).read_bytes(); pos, idat = 8, b""
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8]); body = data[pos + 8:pos + 8 + ln]
        if typ == b"IDAT":
            idat += body
        pos += 12 + ln
    return zlib.decompress(idat)  # raw filtered scanlines: identical encoder + identical pixels -> identical bytes


def render(arm, bindir, mp, a, tag, perf=None):
    png = ART / "png" / f"{tag}.png"; png.parent.mkdir(exist_ok=True)
    log = ART / "logs" / f"{tag}.log"; log.parent.mkdir(exist_ok=True)
    cmd = ["stdbuf", "-oL", "-eL", str(bindir / "sd-cli"), "--diffusion-model", mp["dm"], "--llm", mp["te"], "--vae", mp["vae"],
           "-p", PROMPT, "-W", str(a.size), "-H", str(a.size), "--steps", str(a.steps), "--seed", "42", "--cfg-scale", "1",
           "--sampling-method", "euler", "-t", str(os.cpu_count()), "--backend", "cpu", "--diffusion-fa", "-o", str(png), "-v"]
    say(f"run {tag}: {' '.join(cmd[3:])}")
    t0 = time.time(); fh = open(log, "w")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    pp = pp2 = None; marks = {}
    for line in p.stdout:
        fh.write(line); fh.flush()
        if "generating image:" in line and "start" not in marks:
            marks["start"] = time.time() - t0
            if perf:
                # probe on this runner image: cpu-clock/task-clock refuse dwarf call graphs; hardware cycles accept them
                pp = subprocess.Popen([perf["bin"], "record", "-p", str(p.pid), "-e", "cycles", "-F", str(perf["freq"]),
                                       "--call-graph", "dwarf,8192", "-o", perf["data"]],
                                      stdout=open(ART / f"perf_record_{tag}.log", "w"), stderr=subprocess.STDOUT)
                pp2 = subprocess.Popen([perf["bin"], "record", "-p", str(p.pid), "-e", "cpu-clock", "-F", "199",
                                        "-o", perf["data"] + ".clk"],
                                       stdout=open(ART / f"perf_record_clk_{tag}.log", "w"), stderr=subprocess.STDOUT)
        if "sampling completed" in line and "end" not in marks:
            marks["end"] = time.time() - t0
            if pp:
                pp.send_signal(signal.SIGINT); pp2.send_signal(signal.SIGINT); pp.wait(timeout=600); pp2.wait(timeout=600)
    _, status, ru = os.wait4(p.pid, 0); rc = os.waitstatus_to_exitcode(status); p.returncode = rc
    if pp and pp.poll() is None:
        pp.send_signal(signal.SIGINT); pp.wait(timeout=600)
    fh.close(); wall = time.time() - t0
    text = log.read_text(errors="replace")
    rec = {"tag": tag, "arm": arm, "rc": rc, "wall": round(wall, 2), "peak_rss_mb": round(ru.ru_maxrss / 1024, 1), "marks": marks}
    for k, pat in {"sampling": r"sampling completed, taking ([\d.]+)s", "te": r"get_learned_condition completed, taking ([\d.]+)\s*s",
                   "vae": r"decode_first_stage completed, taking ([\d.]+)\s*s", "generate": r"generate_image completed in ([\d.]+)\s*s"}.items():
        m = re.findall(pat, text)
        if m:
            rec[k] = float(m[-1])
    if "sampling" in rec:
        rec["s_per_step"] = round(rec["sampling"] / a.steps, 3)
    m = re.search(r"loaded CPU backend from (\S+)", text); rec["variant"] = os.path.basename(m.group(1)) if m else None
    rec["fa_lines"] = sorted(set(l.strip()[:200] for l in text.splitlines() if re.search(r"flash|repack|CPU_REPACK", l, re.I)))[:12]
    if rc == 0 and png.exists():
        rec["sha256_file"] = hashlib.sha256(png.read_bytes()).hexdigest()
        rec["sha256_pix"] = hashlib.sha256(png_pixels(png)).hexdigest()
    else:
        rec["tail"] = text[-3000:]
    say("  ->", json.dumps({k: v for k, v in rec.items() if k not in ("fa_lines", "tail")}))
    return rec


FA_RE = re.compile(r"flash_attn_ext")
Q8K_RE = re.compile(r"quantize_mat_q8_K|quantize_row_q8_K|ggml_quantize_mat|quantize_q8_K")
SPIN_RE = re.compile(r"ggml_barrier|gomp_|GOMP_|do_spin|gomp_team_barrier|futex|sched_yield|ggml_graph_compute_thread\b")


def analyse(perf_bin, data, tag):
    """Inclusive / self shares over every sample of the sampling window."""
    out = ART / f"perf_script_{tag}.txt"
    with open(out, "w") as fh:
        subprocess.run([perf_bin, "script", "-i", data, "-F", "tid,time,ip,sym,dso", "--no-inline"], stdout=fh,
                       stderr=open(ART / f"perf_script_{tag}.err", "w"))
    n = fa = q8 = spin = 0; self_sym = collections.Counter(); fa_self = collections.Counter(); q8_self = collections.Counter()
    frames = []

    def flush():
        nonlocal n, fa, q8, spin
        if not frames:
            return
        n += 1
        leaf = frames[0]
        self_sym[leaf] += 1
        if any(FA_RE.search(f) for f in frames):
            fa += 1; fa_self[leaf] += 1
        if any(Q8K_RE.search(f) for f in frames):
            q8 += 1; q8_self[leaf] += 1
        if SPIN_RE.search(leaf):
            spin += 1

    for line in open(out, errors="replace"):
        if not line.strip():
            flush(); frames = []; continue
        if line[0] in " \t":
            parts = line.strip().split(None, 1)
            sym = parts[1] if len(parts) > 1 else "?"
            sym = re.sub(r"\+0x[0-9a-f]+", "", sym)
            frames.append(sym)
        else:
            flush(); frames = []
    flush()
    res = {"samples": n, "fa_incl_pct": round(100 * fa / max(n, 1), 2), "q8k_incl_pct": round(100 * q8 / max(n, 1), 2),
           "spin_leaf_pct": round(100 * spin / max(n, 1), 2),
           "fa_incl_pct_excl_spin": round(100 * fa / max(n - spin, 1), 2), "q8k_incl_pct_excl_spin": round(100 * q8 / max(n - spin, 1), 2),
           "top_self": [(s, c, round(100 * c / max(n, 1), 2)) for s, c in self_sym.most_common(30)],
           "fa_leaf_breakdown": [(s, c, round(100 * c / max(n, 1), 2)) for s, c in fa_self.most_common(12)],
           "q8k_leaf_breakdown": [(s, c, round(100 * c / max(n, 1), 2)) for s, c in q8_self.most_common(6)]}
    subprocess.run(f"{perf_bin} report -i {data} --no-children --sort sym --stdio --percent-limit 0.3 -g none > {ART}/perf_report_self_{tag}.txt 2>&1", shell=True)
    subprocess.run(f"{perf_bin} report -i {data} --children --sort sym --stdio --percent-limit 1 -g none > {ART}/perf_report_children_{tag}.txt 2>&1", shell=True)
    if Path(data + ".clk").exists():
        subprocess.run(f"{perf_bin} report -i {data}.clk --sort sym --stdio --percent-limit 0.2 > {ART}/perf_report_clk_self_{tag}.txt 2>&1", shell=True)
    subprocess.run(["gzip", "-f", str(out)])
    say("perf", tag, json.dumps({k: v for k, v in res.items() if not k.startswith("top")})); say(json.dumps(res["top_self"][:15]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8); ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--gate", type=float, default=3.0); ap.add_argument("--force-ab", action="store_true")
    ap.add_argument("--no-ab", action="store_true"); ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--ref", default="e9c73eccf85dc59981b63ac34a98361dffae7ab5"); ap.add_argument("--freq", type=int, default=49)
    a = ap.parse_args()
    res = {"args": vars(a), "lscpu": sh("lscpu", check=False), "features": re.findall(r"Features\s*:\s*(.*)", Path("/proc/cpuinfo").read_text())[:1],
           "meminfo": Path("/proc/meminfo").read_text().splitlines()[:3], "runs": []}
    srcs = make_src(a.ref)
    bins = {k: build(k, s) for k, s in srcs.items()}
    for k, b in bins.items():
        res[f"syms_{k}"] = sh(f"for f in {b}/libggml-cpu-*.so; do echo $f $(nm -D --defined-only $f | grep -c 'quantize_mat_q8_K'); done; "
                              f"nm -C {b}/libggml-cpu-armv8.2_2.so 2>/dev/null | grep -i 'quantize_mat_q8_K' | head", check=False)
    for v in ("armv8.6_0", "armv8.2_2", "armv8.0_1"):
        lb, lp = bins["A"] / f"libggml-cpu-{v}.so", bins["B"] / f"libggml-cpu-{v}.so"
        if lb.exists():
            exe = W / "qtest"
            sh(["g++-14", "-O1", "-std=c++17", str(H / "qtest.cpp"), "-ldl", "-o", str(exe)], check=False)
            r = subprocess.run([str(exe), str(lb), str(lp)], capture_output=True, text=True)
            res.setdefault("qtest", {})[v] = {"rc": r.returncode, "out": r.stdout[-3000:] + r.stderr[-1000:]}
            say("qtest", v, r.returncode, r.stdout[-1500:])
    mp = fetch_models(a.model)
    perf_bin = (sh("ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1", check=False).strip() or shutil.which("perf") or "perf")
    res["perf_bin"] = perf_bin; res["perf_version"] = sh(f"{perf_bin} --version", check=False).strip()
    data = str(W / "perf_A.data")
    r = render("A", bins["A"], mp, a, f"{a.model}_prof_A", perf={"bin": perf_bin, "freq": a.freq, "data": data})
    res["runs"].append(r)
    res["perf"] = analyse(perf_bin, data, f"{a.model}_A") if Path(data).exists() else {"err": "no perf data"}
    (ART / "results.json").write_text(json.dumps(res, indent=1))
    q8 = res["perf"].get("q8k_incl_pct", 0) or 0
    if a.confirm:
        # one unprofiled render per arm: 8-step sha256 base vs patched + perf-free s/step
        for i, arm in enumerate("AB"):
            res["runs"].append(render(arm, bins[arm], mp, a, f"{a.model}_c{i}{arm}"))
            (ART / "results.json").write_text(json.dumps(res, indent=1))
    res["ab_ran"] = bool(a.force_ab or (q8 >= a.gate and not a.no_ab))
    say(f"q8K share {q8}% gate {a.gate}% -> ABBA {'RUN' if res['ab_ran'] else 'SKIPPED (kill)'}")
    if res["ab_ran"]:
        for rd in range(a.rounds):
            for i, arm in enumerate("ABBA"):
                res["runs"].append(render(arm, bins[arm], mp, a, f"{a.model}_r{rd}_{i}{arm}"))
                (ART / "results.json").write_text(json.dumps(res, indent=1))
        tim = [x for x in res["runs"] if "_r" in x["tag"] and x["rc"] == 0]
        def med(v):
            v = sorted(v); return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
        A = [x["s_per_step"] for x in tim if x["arm"] == "A"]; B = [x["s_per_step"] for x in tim if x["arm"] == "B"]
        s = {"A": A, "B": B, "A_med": med(A) if A else None, "B_med": med(B) if B else None,
             "A_spread_pct": round(100 * (max(A) - min(A)) / med(A), 2) if A else None,
             "B_spread_pct": round(100 * (max(B) - min(B)) / med(B), 2) if B else None,
             "sha_file": sorted({(x["arm"], x.get("sha256_file")) for x in res["runs"]}),
             "sha_pix": sorted({(x["arm"], x.get("sha256_pix")) for x in res["runs"]})}
        if A and B:
            s["speedup_pct"] = round(100 * (s["A_med"] / s["B_med"] - 1), 2)
        s["pix_identical_all"] = len({x.get("sha256_pix") for x in res["runs"] if x["rc"] == 0}) == 1
        res["summary"] = s; say("SUMMARY", json.dumps(s))
    (ART / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
