#!/usr/bin/env python3
"""Tiny random-weight Diffusers pipelines for Studio's multi-GPU matrix.

Real model_index / modular_model_index / scheduler / tokenizer / processor / guider files are copied from the cached
real repo; every model component is rebuilt from its REAL config with shrunk dims and random weights (bf16 denoiser
and text encoders, float32 VAE). No weights are read or downloaded.

Rules that keep the tiny pipelines runnable:
  * decoder-LM text encoders keep their REAL vocab_size and REAL num_hidden_layers (pipelines tap fixed hidden-state
    layers: FLUX.2 (9,18,27) / (10,20,30), Krea-2 (2..35), Ideogram-4 (0..35), FLUX.3); only width / heads shrink.
  * mrope sections are rescaled so they sum to head_dim / 2.
  * the denoiser's text-input width follows the shrunk text encoder (e.g. FLUX.2 joint_attention_dim = 3 * hidden).

usage: python tiny_build.py <family> [<family> ...] | all      (output: temp/mgpu_tiny/<family>)
"""

from __future__ import annotations

import copy
import glob
import json
import os
import shutil
import sys

WS = "/mnt/disks/unslothai/ubuntu/workspace_81"
OUT = os.path.join(WS, "temp", "mgpu_tiny")
HUB = os.path.join(os.environ.get("HF_HOME", "/mnt/disks/unslothai/ubuntu/.cache/huggingface"), "hub")

TXT_H = 64  # text hidden: 2 heads x head_dim 32
MROPE = [4, 6, 6]  # sums to head_dim / 2 = 16 (Qwen2.5-VL / Qwen3-VL text)


def snap(repo, need = "model_index.json"):
    ds = sorted(glob.glob(os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots", "*")),
                key = os.path.getmtime, reverse = True)
    for d in ds:
        if all(os.path.exists(os.path.join(d, n)) for n in need.split(",")):
            return d
    raise SystemExit(f"not cached (with {need}): {repo}")


def copy_meta(src, dst, names):
    """Copy non-weight files (json / txt / jinja / tokenizer models) of the named files/dirs."""
    os.makedirs(dst, exist_ok = True)
    for n in names:
        s = os.path.join(src, n)
        if os.path.isdir(s):
            for root, _dirs, files in os.walk(s, followlinks = True):
                for f in files:
                    if f.endswith((".safetensors", ".bin", ".gguf", ".pt", ".pth")) or f.endswith(".index.json"):
                        continue
                    if f == "config.json" and n.startswith("text_encoder"):
                        continue
                    rel = os.path.relpath(os.path.join(root, f), src)
                    os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok = True)
                    shutil.copy(os.path.join(root, f), os.path.join(dst, rel))
        elif os.path.isfile(s):
            shutil.copy(s, os.path.join(dst, n))


# --------------------------------------------------------------------------------------------------- transformers
def _mrope(c):
    for k in ("rope_scaling", "rope_parameters"):
        r = c.get(k)
        if isinstance(r, dict) and "mrope_section" in r:
            r = dict(r)
            r["mrope_section"] = list(MROPE)
            c[k] = r


def _shrink_text(c, keep_layers = True, layers = 2):
    c = dict(c)
    mt = str(c.get("model_type", ""))
    if mt.startswith("t5") or "d_model" in c:
        c.update(d_model = 32, d_ff = 64, d_kv = 16, num_heads = 2, num_layers = 2, num_decoder_layers = 1)
        return c
    if mt.startswith("clip"):
        c.update(hidden_size = 32, intermediate_size = 64, num_attention_heads = 2, num_hidden_layers = 2,
                 projection_dim = 32)
        return c
    for k, v in (("hidden_size", TXT_H), ("intermediate_size", 2 * TXT_H), ("num_attention_heads", 2),
                 ("num_key_value_heads", 1), ("head_dim", TXT_H // 2)):
        if k in c or k in ("hidden_size",):
            c[k] = v
    if not keep_layers:
        c["num_hidden_layers"] = layers
        if isinstance(c.get("layer_types"), list):
            c["layer_types"] = c["layer_types"][:layers]
        if "max_window_layers" in c:
            c["max_window_layers"] = layers
    _mrope(c)
    return c


def _shrink_vision(v, out_hidden):
    v = dict(v)
    mt = str(v.get("model_type", ""))
    if mt == "pixtral":
        v.update(hidden_size = 32, intermediate_size = 64, num_attention_heads = 2, num_hidden_layers = 1, head_dim = 16)
        return v
    v.update(depth = 1, hidden_size = 32, intermediate_size = 64, num_heads = 2, out_hidden_size = out_hidden)
    if "fullatt_block_indexes" in v:
        v["fullatt_block_indexes"] = [0]
    if "deepstack_visual_indexes" in v:
        v["deepstack_visual_indexes"] = [0]
    if "embed_dim" in v:
        v["embed_dim"] = 32
    return v


def build_tf(src_dir, dst_dir, keep_layers = True, layers = 2):
    import torch
    import transformers

    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    arch = cfg["architectures"][0]
    small = copy.deepcopy(cfg)
    for k in ("quantization_config", "ideogram_fp8_weight_only"):
        small.pop(k, None)
    if "text_config" in small or "vision_config" in small:
        if isinstance(small.get("text_config"), dict):
            small["text_config"] = _shrink_text(small["text_config"], keep_layers, layers)
        if "hidden_size" in small:  # Qwen2.5-VL repeats the text fields at the top level
            small = {**small, **{k: v for k, v in _shrink_text(small, keep_layers, layers).items()
                                 if k not in ("text_config", "vision_config")}}
        if isinstance(small.get("vision_config"), dict):
            small["vision_config"] = _shrink_vision(small["vision_config"], TXT_H)
    else:
        small = _shrink_text(small, keep_layers, layers)
    os.makedirs(dst_dir, exist_ok = True)
    json.dump(small, open(os.path.join(dst_dir, "config.json"), "w"), indent = 2)
    conf = transformers.AutoConfig.from_pretrained(dst_dir)
    cls = getattr(transformers, arch)
    torch.manual_seed(0)
    m = cls(conf).to(torch.bfloat16).eval()
    m.save_pretrained(dst_dir)
    for f in os.listdir(src_dir):  # generation_config etc.
        if f.endswith(".json") and f not in ("config.json",) and not f.endswith(".index.json") \
                and not os.path.exists(os.path.join(dst_dir, f)):
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    print("built", arch, sum(p.numel() for p in m.parameters()), "layers",
          getattr(getattr(conf, "text_config", conf), "num_hidden_layers", None))


# ------------------------------------------------------------------------------------------------------ diffusers
def build_df(src_dir, dst_dir, overrides, dtype = "bfloat16"):
    import diffusers
    import torch

    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    for k in ("quantization_config", "_name_or_path"):
        cfg.pop(k, None)
    cls = getattr(diffusers, cfg["_class_name"])
    cfg.update(overrides)
    torch.manual_seed(0)
    m = cls.from_config(cfg).to(getattr(torch, dtype))
    m.save_pretrained(dst_dir)
    print("built", cfg["_class_name"], sum(p.numel() for p in m.parameters()), dtype)


VAE_KL = dict(block_out_channels = [32, 32, 32, 32], layers_per_block = 1)
VAE_QWEN = dict(base_dim = 32, num_res_blocks = 1)


def standard(fam, repo, te, df, meta, keep_layers = True):
    """te: {component: None}; df: {component: (overrides, dtype)}."""
    s = snap(repo)
    out = os.path.join(OUT, fam)
    os.makedirs(out, exist_ok = True)
    copy_meta(s, out, meta)
    for c in te:
        build_tf(os.path.join(s, c), os.path.join(out, c), keep_layers = keep_layers)
    for c, (ov, dt) in df.items():
        build_df(os.path.join(s, c), os.path.join(out, c), ov, dt)
    return out


def flux2_klein():
    return standard("flux.2-klein", "black-forest-labs/FLUX.2-klein-4B", ["text_encoder"], {
        "transformer": (dict(num_layers = 1, num_single_layers = 1, attention_head_dim = 32, num_attention_heads = 2,
                             axes_dims_rope = [8, 8, 8, 8], joint_attention_dim = 3 * TXT_H), "bfloat16"),
        "vae": (VAE_KL, "float32")}, ["model_index.json", "scheduler", "tokenizer"])


def flux2_dev():
    return standard("flux.2-dev", "unsloth/FLUX.2-dev", ["text_encoder"], {
        "transformer": (dict(num_layers = 1, num_single_layers = 1, attention_head_dim = 32, num_attention_heads = 2,
                             axes_dims_rope = [8, 8, 8, 8], joint_attention_dim = 3 * TXT_H), "bfloat16"),
        "vae": (VAE_KL, "float32")}, ["model_index.json", "scheduler", "tokenizer"])


def hidream():
    import torch
    import transformers

    out = standard("hidream-i1", "unsloth/HiDream-I1-Full", ["text_encoder", "text_encoder_2", "text_encoder_3"], {
        "transformer": (dict(num_layers = 1, num_single_layers = 1, attention_head_dim = 32, num_attention_heads = 2,
                             axes_dims_rope = [16, 8, 8], caption_channels = [32, TXT_H], text_emb_dim = 64,
                             llama_layers = [0, 31]), "bfloat16"),
        "vae": (VAE_KL, "float32")},
        ["model_index.json", "scheduler", "tokenizer", "tokenizer_2", "tokenizer_3"])
    # text_encoder_4 / tokenizer_4: the repo names them but does not ship them; Studio loads
    # unsloth/Meta-Llama-3.1-8B-Instruct. Ship a tiny copy of that REAL Llama config (32 layers kept) + real tokenizer.
    ls = snap("unsloth/Meta-Llama-3.1-8B-Instruct", "config.json,tokenizer.json")
    copy_meta(ls, os.path.join(out, "tokenizer_4"), ["tokenizer.json", "tokenizer_config.json",
                                                     "special_tokens_map.json"])
    build_tf(ls, os.path.join(out, "text_encoder_4"), keep_layers = True)
    return out


def krea2():
    return standard("krea-2", "unsloth/Krea-2-Turbo", ["text_encoder"], {
        "transformer": (dict(attention_head_dim = 32, num_attention_heads = 2, num_key_value_heads = 1,
                             axes_dims_rope = [8, 12, 12], intermediate_size = 128, num_layers = 1,
                             num_layerwise_text_blocks = 1, num_refiner_text_blocks = 1, text_hidden_dim = TXT_H,
                             text_intermediate_size = 128, text_num_attention_heads = 2,
                             text_num_key_value_heads = 2), "bfloat16"),
        "vae": (VAE_QWEN, "float32")}, ["model_index.json", "scheduler", "tokenizer"])


def ideogram4():
    tov = dict(attention_head_dim = 64, num_attention_heads = 2, intermediate_size = 128, num_layers = 1,
               adaln_dim = 32, llm_features_dim = 13 * TXT_H, mrope_section = [6, 5, 5])
    return standard("ideogram-4", "unsloth/ideogram-4-fp8", ["text_encoder"], {
        "transformer": (tov, "bfloat16"), "unconditional_transformer": (tov, "bfloat16"),
        "vae": (VAE_KL, "float32")}, ["model_index.json", "scheduler", "tokenizer"])


def hunyuanimage():
    return standard("hunyuanimage-2.1", "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
                    ["text_encoder", "text_encoder_2"], {
        "transformer": (dict(attention_head_dim = 32, num_attention_heads = 2, num_layers = 1, num_refiner_layers = 1,
                             num_single_layers = 1, rope_axes_dim = [16, 16], text_embed_dim = TXT_H,
                             text_embed_2_dim = 32), "bfloat16"),
        "vae": (dict(block_out_channels = [32, 32, 32, 32, 128, 128], layers_per_block = 1), "float32")},
        ["model_index.json", "scheduler", "tokenizer", "tokenizer_2", "guider", "ocr_guider"])


def _qwen(fam, repo):
    return standard(fam, repo, ["text_encoder"], {
        "transformer": (dict(attention_head_dim = 32, num_attention_heads = 2, axes_dims_rope = [8, 12, 12],
                             joint_attention_dim = TXT_H, num_layers = 1), "bfloat16"),
        "vae": (VAE_QWEN, "float32")}, ["model_index.json", "scheduler", "tokenizer", "processor"])


def qwen_layered():
    return _qwen("qwen-image-layered", "Qwen/Qwen-Image-Layered")


def qwen_edit():
    return _qwen("qwen-image-edit", "unsloth/Qwen-Image-Edit-2511")


def _flux3(fam, repo):
    s = snap(repo, "modular_model_index.json,text_encoder/config.json,transformer/config.json,vae/config.json")
    out = os.path.join(OUT, fam)
    os.makedirs(out, exist_ok = True)
    copy_meta(s, out, ["modular_model_index.json", "text_encoder"])
    build_tf(os.path.join(s, "text_encoder"), os.path.join(out, "text_encoder"), keep_layers = True)
    tcfg = json.load(open(os.path.join(s, "transformer", "config.json")))
    te_cfg = json.load(open(os.path.join(s, "text_encoder", "config.json")))
    n_tap = tcfg["joint_attention_dim"] // te_cfg["text_config"]["hidden_size"]
    build_df(os.path.join(s, "transformer"), os.path.join(out, "transformer"), dict(
        num_layers = 1, num_single_layers = 1, attention_head_dim = 32, num_attention_heads = 2,
        axes_dims_rope = [8, 8, 8, 8], joint_attention_dim = n_tap * TXT_H), "bfloat16")
    build_df(os.path.join(s, "vae"), os.path.join(out, "vae"), VAE_KL, "float32")
    p = os.path.join(out, "modular_model_index.json")
    index = json.load(open(p))
    for spec in index.values():
        if isinstance(spec, list) and len(spec) >= 3 and isinstance(spec[2], dict):
            spec[2]["pretrained_model_name_or_path"] = out
    json.dump(index, open(p, "w"), indent = 2)
    return out


def flux3_klein():
    return _flux3("flux.3-klein", "black-forest-labs/flux-3-image-klein")


def flux3_klein_base():
    return _flux3("flux.3-klein-base", "black-forest-labs/flux-3-image-klein-base")


def hv15():
    s = snap("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    out = os.path.join(OUT, "hunyuanvideo-1.5")
    os.makedirs(out, exist_ok = True)
    copy_meta(s, out, ["model_index.json", "scheduler", "tokenizer", "tokenizer_2", "guider"])
    build_tf(os.path.join(s, "text_encoder"), os.path.join(out, "text_encoder"), keep_layers = True)
    build_tf(os.path.join(s, "text_encoder_2"), os.path.join(out, "text_encoder_2"), keep_layers = True)
    build_df(os.path.join(s, "transformer"), os.path.join(out, "transformer"), dict(
        attention_head_dim = 64, num_attention_heads = 2, num_layers = 2, num_refiner_layers = 1,
        rope_axes_dim = [16, 24, 24], text_embed_dim = TXT_H, text_embed_2_dim = 32, image_embed_dim = 32,
        mlp_ratio = 2.0), "bfloat16")
    build_df(os.path.join(s, "vae"), os.path.join(out, "vae"), dict(
        block_out_channels = [16, 16, 32, 32, 32], layers_per_block = 1), "float32")
    return out


def minimax_h3():
    s = snap("MiniMaxAI/MiniMax-H3", "modular_model_index.json,transformer/config.json")
    out = os.path.join(OUT, "minimax-h3")
    os.makedirs(out, exist_ok = True)
    copy_meta(s, out, ["modular_model_index.json", "model_index.json", "processor", "tokenizer", "scheduler",
                       "audio_scheduler", "text_encoder"])
    build_tf(os.path.join(s, "text_encoder"), os.path.join(out, "text_encoder"), keep_layers = True)
    # rope rotates 2 * 3 * rope_freq_dim of head_dim (real 96 of 128): keep that ratio, 48 of 64.
    tov = dict(num_attention_heads = 2, attention_head_dim = 64, hidden_size = 128, num_layers = 1,
               num_refiner_layers = 1, ffn_dim = 256, text_dim = TXT_H, time_embed_hidden_dim = 128,
               time_embed_dim = 64, rope_freq_dim = 8)
    for t in ("transformer", "transformer_ref"):
        build_df(os.path.join(s, t), os.path.join(out, t), tov, "bfloat16")
    build_df(os.path.join(s, "vae"), os.path.join(out, "vae"), dict(
        block_out_channels = [32] * 6, layers_per_block = 1, decoder_num_layers = 1, decoder_num_attention_heads = 2,
        decoder_attention_head_dim = 32, decoder_ffn_mult = 2), "float32")
    build_df(os.path.join(s, "audio_vae"), os.path.join(out, "audio_vae"), dict(
        encoder_dim = 16, latent_dim = 64, decoder_dim = 256, num_attention_heads = 2), "float32")
    for name in ("modular_model_index.json", "model_index.json"):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            continue
        index = json.load(open(p))
        for spec in index.values():
            if isinstance(spec, list) and len(spec) >= 3 and isinstance(spec[2], dict):
                spec[2]["pretrained_model_name_or_path"] = out
        json.dump(index, open(p, "w"), indent = 2)
    return out


FAMS = {
    "minimax-h3": minimax_h3,
    "flux.2-klein": flux2_klein, "flux.2-dev": flux2_dev, "hidream-i1": hidream, "krea-2": krea2,
    "flux.3-klein": flux3_klein, "flux.3-klein-base": flux3_klein_base, "ideogram-4": ideogram4,
    "hunyuanimage-2.1": hunyuanimage, "qwen-image-layered": qwen_layered, "qwen-image-edit": qwen_edit,
    "hunyuanvideo-1.5": hv15,
}


def main():
    names = sys.argv[1:]
    if names == ["all"]:
        names = list(FAMS)
    for n in names:
        out = os.path.join(OUT, n)
        if os.path.isdir(out):
            shutil.rmtree(os.path.join(OUT, n))
        print("==", n)
        print("OK", FAMS[n]())


if __name__ == "__main__":
    main()
