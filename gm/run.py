"""GGUF weight-type matrix driver (one staging job = one leg x one shard).

Builds each ref of the plan (fork master, the CPU PR stack head, optional kernel branches) like the prebuilt leg
(ggml-patches applied in glob order, same cmake defines, all CPU variants where the leg ships them), plus
test-backend-ops and qmm_bench (op-level MUL_MAT per weight type: time, path, error vs use_ref and an fp64
oracle). Shards: "op" = qmm_bench sweeps (+ test-backend-ops MUL_MAT on a GPU backend), "d*" = FLUX.2-klein-4B DiT
types at 512 px, "te" = Qwen3-4B text encoder types. Arms interleave refs per rep (A/A spread). Results in ./art.
"""
import argparse, hashlib, json, os, platform, re, shutil, subprocess, sys, time, zlib, struct
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = Path.cwd()
W = ROOT / "w"; W.mkdir(exist_ok=True)
ART = ROOT / "art"; ART.mkdir(exist_ok=True)
FORK = "https://github.com/unslothai/stable-diffusion.cpp"
IS_MAC = sys.platform == "darwin"
HF = "https://huggingface.co"
KLEIN = HF + "/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-{t}.gguf"
TE = HF + "/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-{t}.gguf"
VAE = HF + "/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors"
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
LOG = open(ART / "driver.log", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def sh(cmd, log=None, cwd=None, env=None, check=True, timeout=None):
    say("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))
    with open(log or os.devnull, "a", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env, timeout=timeout,
                           stdout=fh if log else subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    out = "" if log else (p.stdout or "")
    if check and p.returncode != 0:
        tail = Path(log).read_text(errors="replace")[-4000:] if log else out[-4000:]
        raise RuntimeError(f"rc={p.returncode}: {cmd}\n{tail}")
    return out


def host_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "nproc": os.cpu_count()}
    try:
        if IS_MAC:
            info["cpu"] = sh("sysctl -n machdep.cpu.brand_string", check=False).strip()
            info["mem"] = sh("sysctl -n hw.memsize", check=False).strip()
            info["features"] = sh("sysctl -a | grep -E 'hw.optional.(arm|avx|AVX)' | grep ': 1'", check=False)[:3000]
        else:
            txt = Path("/proc/cpuinfo").read_text()
            info["cpu"] = next((l.split(":", 1)[1].strip() for l in sh("lscpu", check=False).splitlines() if "Model name" in l), "")
            info["x86_flags"] = sorted(set(re.findall(r"\b(avx512\w*|avx_vnni|amx_\w+|avx2|f16c|fma)\b", txt)))
            info["arm_features"] = sorted(set(re.findall(r"\b(asimddp|i8mm|sve2?|sme\w*|fphp|asimdhp|bf16)\b", txt)))
            info["mem"] = next((l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")), "")
    except Exception as e:  # noqa: BLE001
        info["err"] = str(e)
    return info


def make_src(fork, name, spec):
    src = W / f"src_{name}"
    if src.exists():
        return src
    first = f"refs/pr/{spec['pr']}" if "pr" in spec else spec.get("ref", "origin/master")
    sh(["git", "-C", str(fork), "worktree", "add", "-q", "--detach", str(src), first])
    sh(["git", "-C", str(src), "submodule", "update", "--init", "--depth", "1", "ggml"])
    patches = sorted((src / "scripts/unsloth/ggml-patches").glob("*.patch"))
    for p in patches:
        sh(["git", "-C", str(src / "ggml"), "apply", f"../scripts/unsloth/ggml-patches/{p.name}"], log=str(ART / f"patch_{name}.log"))
    # qmm_bench joins the ggml tests (built with GGML_BUILD_TESTS=ON next to test-backend-ops)
    shutil.copy(H / "qmm_bench.cpp", src / "ggml/tests/qmm_bench.cpp")
    with open(src / "ggml/tests/CMakeLists.txt", "a") as f:
        f.write("\nadd_executable(qmm_bench qmm_bench.cpp)\ntarget_link_libraries(qmm_bench PRIVATE ggml)\n")
    meta = {"name": name, "spec": spec, "head": sh(["git", "-C", str(src), "rev-parse", "HEAD"]).strip(),
            "ggml": sh(["git", "-C", str(src / "ggml"), "rev-parse", "HEAD"]).strip(), "patches": [p.name for p in patches]}
    (ART / f"src_{name}.json").write_text(json.dumps(meta, indent=1))
    say("source", name, meta["head"], len(patches), "patches")
    return src


def build(name, src, leg):
    bd = src / "build"; log = str(ART / f"build_{name}.log")
    defs = ["-DCMAKE_BUILD_TYPE=Release", "-DSD_BUILD_EXAMPLES=ON", "-DSD_SERVER_BUILD_FRONTEND=OFF", "-DSD_WEBP=OFF",
            "-DSD_WEBM=OFF", "-DGGML_NATIVE=OFF", "-DGGML_BUILD_TESTS=ON"]
    if leg.get("variants"):
        defs += ["-DSD_BUILD_SHARED_GGML_LIB=ON", "-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON"]
        if IS_MAC:
            defs += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=@loader_path"]
    defs += leg.get("defines", [])
    if shutil.which("ccache"):
        defs += ["-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]
    t = time.time()
    try:
        sh(["cmake", "-S", str(src), "-B", str(bd), "-G", "Ninja"] + defs, log=log)
        sh(["cmake", "--build", str(bd), "-j", str(os.cpu_count()), "--target", "sd-cli", "test-backend-ops", "qmm_bench"], log=log)
        rc = 0
    except RuntimeError as e:
        rc = 1; say("BUILD FAILED", name, str(e)[-3000:])
    say(f"built {name} rc={rc} in {time.time() - t:.0f}s")
    return {"rc": rc, "seconds": round(time.time() - t), "bin": str(bd / "bin")}


def pinned(bindir, variant, tag):
    d = W / f"pin_{tag}_{variant}"
    if not d.exists():
        shutil.copytree(bindir, d, symlinks=True)
        for f in d.iterdir():
            if "ggml-cpu-" in f.name and variant not in f.name:
                f.unlink()
    return d


def png_pixels(path):
    data = Path(path).read_bytes(); pos, idat = 8, b""
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8]); body = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif typ == b"IDAT":
            idat += body
        pos += 12 + ln
    return zlib.decompress(idat)  # filtered scanlines: fine for an md5 of the pixels


def fetch(url, dst):
    if dst.exists() and dst.stat().st_size > 0:
        return True
    for i in range(4):
        if subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(dst), url]).returncode == 0:
            return True
        time.sleep(15)
    return False


PAT = {"te": r"get_learned_condition completed, taking ([\d.]+)\s*s", "sampling": r"sampling completed, taking ([\d.]+)\s*s",
       "vae": r"decode_first_stage completed, taking ([\d.]+)\s*s", "load": r"loading tensors completed, taking ([\d.]+)\s*s"}


def run_sd(cli, args, env, tag, timeout):
    log = ART / "logs" / f"{tag}.log"; log.parent.mkdir(exist_ok=True)
    code = ("import resource,subprocess,sys;r=subprocess.call(sys.argv[1:]);"
            "m=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;"
            "print('MAXRSS_MB=%.1f'%(m/1048576 if sys.platform=='darwin' else m/1024),flush=True);sys.exit(r)")
    cmd = [sys.executable, "-c", code, str(cli)] + args
    say("run", tag, " ".join(args), env)
    t = time.time()
    with open(log, "w") as fh:
        try:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, **env), timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -999
    text = log.read_text(errors="replace")
    rec = {"tag": tag, "rc": rc, "wall": round(time.time() - t, 2), "env": env}
    for k, p in PAT.items():
        m = re.findall(p, text)
        if m:
            rec[k] = float(m[-1])
    m = re.search(r"MAXRSS_MB=([\d.]+)", text); rec["rss_mb"] = float(m.group(1)) if m else None
    m = re.search(r"loaded CPU backend from (\S+)", text); rec["variant"] = os.path.basename(m.group(1)) if m else None
    rec["paths"] = sorted(set(re.findall(r"(repack: repack tensor\s+\S* ?with \w+|amx repack tensor\s+\S* ?of type \w+|ggml-cpu iqk gemm: first \w+)", text)))[:20]
    if rc != 0:
        rec["tail"] = text[-2500:]
    say(f"  -> rc={rc} sampling={rec.get('sampling')} te={rec.get('te')} vae={rec.get('vae')} rss={rec.get('rss_mb')}")
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--leg", required=True); ap.add_argument("--shard", required=True)
    a = ap.parse_args()
    plan = json.loads((H / "plan.json").read_text())
    leg = plan["legs"][a.leg]; shard = plan["shards"][a.shard]
    res = {"leg": a.leg, "shard": a.shard, "host": host_info(), "builds": {}, "qmm": [], "runs": [], "tbo": {}}
    say(json.dumps(res["host"], indent=1)[:2500])
    refs = {k: v for k, v in plan["refs"].items() if k in shard.get("refs", plan["refs"].keys())}
    fork = W / "fork"
    if not fork.exists():
        sh(["git", "clone", "-q", FORK, str(fork)])
    for v in refs.values():
        if "pr" in v:
            sh(["git", "-C", str(fork), "fetch", "-q", "origin", f"+refs/pull/{v['pr']}/head:refs/pr/{v['pr']}"])
        if "branch" in v:
            sh(["git", "-C", str(fork), "fetch", "-q", v.get("remote", FORK), f"+refs/heads/{v['branch']}:refs/pr/{v['branch']}"])
            v["ref"] = f"refs/pr/{v['branch']}"
    bins = {}
    for name, spec in refs.items():
        try:
            src = make_src(fork, name, spec)
        except Exception as e:  # noqa: BLE001
            say("SOURCE FAILED", name, e); res["builds"][name] = {"rc": 2, "err": str(e)[-2000:]}; continue
        b = build(name, src, leg); res["builds"][name] = b
        if b["rc"] == 0:
            bins[name] = Path(b["bin"])
    (ART / "results.json").write_text(json.dumps(res, indent=1))
    nthr = str(leg.get("threads") or os.cpu_count())
    reps = shard.get("reps", 3)

    if shard["kind"] == "op":
        variants = [None] + [v for v in leg.get("pin_variants", [])]
        for rep in range(reps):
            for var in variants:
                for name, bd in bins.items():
                    d = bd
                    if var:
                        if not any(("ggml-cpu-" + var) in f.name for f in bd.iterdir()):
                            continue
                        d = pinned(bd, var, name)
                    for env in shard.get("envs", [{}]):
                        tag = f"qmm_{name}_{var or 'auto'}_{'_'.join(f'{k}={v}' for k, v in env.items()) or 'default'}_r{rep}"
                        cmd = [str(d / "qmm_bench"), "-t", nthr, "-r", str(shard.get("inner", 3))]
                        for s in shard["shapes"]:
                            cmd += ["-s", s]
                        cmd += ["-T", ",".join(shard["types"])]
                        log = ART / "logs" / f"{tag}.log"; log.parent.mkdir(exist_ok=True)
                        say("qmm", tag)
                        p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, GGML_CPU_IQK_DEBUG="1", **env), timeout=5400)
                        log.write_text(p.stdout + "\n--- stderr\n" + p.stderr)
                        lib = re.search(r"loaded CPU backend from (\S+)", p.stderr)
                        for line in p.stdout.splitlines():
                            if line.startswith("{"):
                                r = json.loads(line); r.update({"ref": name, "variant": var or (os.path.basename(lib.group(1)) if lib else None),
                                                                 "env": env, "rep": rep}); res["qmm"].append(r)
                        say(f"  rc={p.returncode} rows={sum(1 for l in p.stdout.splitlines() if l.startswith('{'))}")
                        (ART / "results.json").write_text(json.dumps(res, indent=1))
        for spec in shard.get("tbo", {}).get(a.leg, []):
            for name, bd in bins.items():
                exe = bd / "test-backend-ops"
                mode, be, op = spec.split(":")
                log = ART / f"tbo_{name}_{mode}_{be}_{op}.log"
                say("tbo", name, spec)
                r = subprocess.run([str(exe), mode, "-b", be, "-o", op], stdout=open(log, "w"), stderr=subprocess.STDOUT, timeout=5400)
                text = log.read_text(errors="replace")
                m = re.search(r"(\d+)/(\d+) tests passed", text)
                res["tbo"][f"{name}:{spec}"] = {"rc": r.returncode, "passed": m.group(0) if m else None,
                                                "fail_lines": [l for l in text.splitlines() if "FAIL" in l][:40]}
                say("  ", res["tbo"][f"{name}:{spec}"]["passed"], r.returncode)
    else:
        mdir = W / "models"; mdir.mkdir(exist_ok=True)
        vae = mdir / "vae.safetensors"; fetch(VAE, vae)
        fixed_te = mdir / f"te_{shard.get('te', 'Q4_K_M')}.gguf"
        fixed_dm = mdir / f"dm_{shard.get('dm', 'Q4_0')}.gguf"
        outdir = W / "out"; outdir.mkdir(exist_ok=True)
        kind = shard["kind"]
        if kind == "dit":
            fetch(TE.format(t=shard.get("te", "Q4_K_M")), fixed_te)
        else:
            fetch(KLEIN.format(t=shard.get("dm", "Q4_0")), fixed_dm)
        for t in shard["types"]:
            f = mdir / f"{kind}_{t}.gguf"
            ok = fetch((KLEIN if kind == "dit" else TE).format(t=t), f)
            if not ok:
                res["runs"].append({"tag": f"{t}", "rc": -1, "err": "download failed"}); continue
            timed_out = set()
            for rep in range(reps):
                for name, bd in bins.items():
                    for arm in shard.get("arms", [{"id": "d", "env": {}}]):
                        if arm.get("legs") and a.leg not in arm["legs"]:
                            continue
                        if (name, arm["id"]) in timed_out:
                            continue  # one timeout is the measurement (s/step > timeout / steps); skip the other reps
                        tag = f"{kind}_{t}_{name}_{arm['id']}_r{rep}"
                        png = outdir / f"{tag}.png"
                        dm, te = (f, fixed_te) if kind == "dit" else (fixed_dm, f)
                        args = ["--diffusion-model", str(dm), "--llm", str(te), "--vae", str(vae), "-p", PROMPT,
                                "-W", str(shard.get("size", 512)), "-H", str(shard.get("size", 512)), "--steps", str(shard.get("steps", 2)),
                                "--seed", "42", "--cfg-scale", "1", "--sampling-method", "euler", "-t", nthr, "-o", str(png), "-v"]
                        if not arm.get("gpu"):
                            args += ["--backend", "cpu"]
                        args += arm.get("args", [])
                        rec = run_sd(bd / "sd-cli", args, dict(arm.get("env", {}), GGML_CPU_IQK_DEBUG="1"), tag, shard.get("timeout", 2400))
                        rec.update({"type": t, "kind": kind, "ref": name, "arm": arm["id"], "rep": rep})
                        if rec["rc"] == -999:
                            timed_out.add((name, arm["id"]))
                        if rec["rc"] == 0 and png.exists():
                            rec["md5"] = hashlib.md5(png_pixels(png)).hexdigest()
                        res["runs"].append(rec)
                        (ART / "results.json").write_text(json.dumps(res, indent=1))
            if shard.get("delete", True):
                f.unlink(missing_ok=True)
    (ART / "results.json").write_text(json.dumps(res, indent=1))
    say("SUMMARY builds", {k: v.get("rc") for k, v in res["builds"].items()}, "failed runs",
        sorted({r["tag"] for r in res["runs"] if r.get("rc")}))


if __name__ == "__main__":
    main()
