#!/usr/bin/env python3
"""Tiny random-weight copies of the unsloth/* diffusers packs Studio loads, from their REAL configs.

Configs, tokenizers and scheduler files are fetched from the public unsloth/* repos (json / jinja / txt / py only,
no weights); every model component is rebuilt from its real config with shrunk widths and random weights.

  build_tiny.py --out DIR [--cfg DIR] {krea-2,flux.3-klein,moss-image,ltx-2.5} ...

Needs diffusers (Studio's pin) and transformers >= 5.10 for the LTX-2.5 Gemma 4 encoder.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_build as tb  # noqa: E402

REPOS = {
    "krea-2": "unsloth/Krea-2-Turbo",
    "flux.3-klein": "unsloth/flux-3-image-klein-FP8",
    "moss-image": "unsloth/MOSS-Image-SFT-FP8",
    "ltx-2.5": "unsloth/LTX-2.5-FP8",
}
TXT_H = tb.TXT_H


def fetch_cfg(repo, cfg_root):
    from huggingface_hub import snapshot_download

    d = os.path.join(cfg_root, repo.replace("/", "__"))
    snapshot_download(repo, local_dir = d, allow_patterns = ["*.json", "*.jinja", "*.txt", "*.py"],
                      ignore_patterns = ["samples/*", "images/*"])
    return d


def build_df_cls(cls, src_cfg, dst_dir, overrides, dtype = "bfloat16", drop = ()):
    import torch

    cfg = json.load(open(src_cfg))
    for k in ("quantization_config", "_name_or_path", *drop):
        cfg.pop(k, None)
    cfg.update(overrides)
    torch.manual_seed(0)
    m = cls.from_config(cfg).to(getattr(torch, dtype))
    m.save_pretrained(dst_dir)
    print("built", cls.__name__, sum(p.numel() for p in m.parameters()), dtype, flush = True)
    return m


def krea2(src, out):
    tb.snap = lambda repo, need = "": src
    tb.OUT = os.path.dirname(out)
    return tb.krea2()


def flux3(src, out):
    tb.snap = lambda repo, need = "": src
    tb.OUT = os.path.dirname(out)
    d = tb._flux3("flux.3-klein", "unsloth/flux-3-image-klein-FP8")
    # the index points at the build dir; the runner rewrites it to wherever the pack lands (see fix_index)
    return d


def fix_index(d):
    """Point a modular index's per-component pretrained_model_name_or_path at this folder."""
    for name in ("modular_model_index.json",):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        index = json.load(open(p))
        for spec in index.values():
            if isinstance(spec, list) and len(spec) >= 3 and isinstance(spec[2], dict):
                spec[2]["pretrained_model_name_or_path"] = os.path.abspath(d)
        json.dump(index, open(p, "w"), indent = 2)


def _shrink_qwen35_text(c):
    c = tb._shrink_text(c, keep_layers = True)
    c.update(linear_num_key_heads = 2, linear_num_value_heads = 2, linear_key_head_dim = 16,
             linear_value_head_dim = 16, head_dim = 32, num_attention_heads = 2, num_key_value_heads = 1)
    rp = dict(c.get("rope_parameters") or {})
    # partial_rotary_factor 0.25 of head_dim 32 -> 8 rotary dims -> mrope sections sum to 4
    if "mrope_section" in rp:
        rp["mrope_section"] = [2, 1, 1]
    c["rope_parameters"] = rp
    return c


def moss(src, out, backend):
    import torch
    import transformers

    tb.copy_meta(src, out, ["model_index.json", "scheduler", "tokenizer", "pipeline_moss_image.py", "transformer"])
    # text encoder: Qwen3_5Model, real vocab / layer count, narrow width
    cfg = json.load(open(os.path.join(src, "text_encoder", "config.json")))
    small = copy.deepcopy(cfg)
    small["text_config"] = _shrink_qwen35_text(small["text_config"])
    if isinstance(small.get("vision_config"), dict):
        small["vision_config"] = tb._shrink_vision(small["vision_config"], TXT_H)
    te = os.path.join(out, "text_encoder")
    os.makedirs(te, exist_ok = True)
    json.dump(small, open(os.path.join(te, "config.json"), "w"), indent = 2)
    conf = transformers.AutoConfig.from_pretrained(te)
    torch.manual_seed(0)
    m = getattr(transformers, cfg["architectures"][0])(conf).to(torch.bfloat16).eval()
    m.save_pretrained(te)
    print("built", cfg["architectures"][0], sum(p.numel() for p in m.parameters()), flush = True)
    # transformer: the repo's own class (Studio's in-tree port keeps its state-dict names and config keys)
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", os.path.join(src, "pipeline_moss_image.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    MOSSImageTransformer = mod.MOSSImageTransformer

    build_df_cls(MOSSImageTransformer, os.path.join(src, "transformer", "config.json"), os.path.join(out, "transformer"),
                 dict(dim = 128, ffn_dim = 256, num_heads = 2, num_layers = 2, text_dim = TXT_H))
    import diffusers

    build_df_cls(diffusers.AutoencoderKLFlux2, os.path.join(src, "vae", "config.json"), os.path.join(out, "vae"),
                 dict(block_out_channels = [32, 32, 32, 32], decoder_block_out_channels = [16, 16, 16, 16],
                      layers_per_block = 1, norm_num_groups = 8), dtype = "float32")
    return out


def ltx25(src, out):
    import torch
    import transformers
    from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2 import LTX2DurationHead, LTX2LatentUpsamplerModel, LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    tb.copy_meta(src, out, ["model_index.json", "scheduler", "tokenizer"])
    # --- Gemma 4 unified encoder: real vocab and all 48 layers (connectors read layers + 1 hidden states), narrow
    tcfg = json.load(open(os.path.join(src, "text_encoder", "config.json")))
    t = copy.deepcopy(tcfg["text_config"])
    nl = t["num_hidden_layers"]
    t.update(hidden_size = 32, intermediate_size = 64, num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16,
             global_head_dim = 16, num_global_key_value_heads = 1)
    small = {k: v for k, v in tcfg.items() if k not in ("text_config", "vision_config", "audio_config",
                                                        "ltx_source_checkpoint")}
    small.update(text_config = t, vision_config = None, audio_config = None)
    te = os.path.join(out, "text_encoder")
    os.makedirs(te, exist_ok = True)
    json.dump(small, open(os.path.join(te, "config.json"), "w"), indent = 2)
    shutil.copy(os.path.join(src, "text_encoder", "generation_config.json"), te)
    conf = transformers.AutoConfig.from_pretrained(te)
    torch.manual_seed(0)
    m = transformers.Gemma4UnifiedForConditionalGeneration(conf).to(torch.bfloat16).eval()
    m.save_pretrained(te)
    print("built Gemma4UnifiedForConditionalGeneration", sum(p.numel() for p in m.parameters()), "layers", nl)
    H, lat = 32, 8
    j = lambda c: os.path.join(src, c, "config.json")  # noqa: E731
    o = lambda c: os.path.join(out, c)  # noqa: E731
    tr = build_df_cls(LTX2VideoTransformer3DModel, j("transformer"), o("transformer"), dict(
        in_channels = lat, out_channels = lat, num_attention_heads = 2, attention_head_dim = 16, cross_attention_dim = 32,
        audio_in_channels = 4, audio_out_channels = 4, audio_num_attention_heads = 2, audio_attention_head_dim = 16,
        audio_cross_attention_dim = 32, num_layers = 2, caption_channels = H))
    if getattr(tr, "keyframes_abs_pos_embedding", None) is not None:
        torch.nn.init.normal_(tr.keyframes_abs_pos_embedding, std = 0.1)
        tr.save_pretrained(o("transformer"))
    build_df_cls(LTX2TextConnectors, j("connectors"), o("connectors"), dict(
        caption_channels = H, text_proj_in_factor = nl + 1, video_connector_num_attention_heads = 2,
        video_connector_attention_head_dim = 16, video_connector_num_layers = 1, video_connector_num_learnable_registers = 8,
        audio_connector_num_attention_heads = 2, audio_connector_attention_head_dim = 16, audio_connector_num_layers = 1,
        audio_connector_num_learnable_registers = 8, video_hidden_dim = 32, audio_hidden_dim = 32))
    build_df_cls(AutoencoderKLLTX2Video, j("vae"), o("vae"), dict(
        latent_channels = lat, block_out_channels = [8, 16, 32, 32], decoder_block_out_channels = [8, 16, 16, 32],
        layers_per_block = [1, 1, 1, 1, 1], decoder_layers_per_block = [1, 1, 1, 1, 1]))
    av = build_df_cls(AutoencoderKLLTX2Audio, j("audio_vae"), o("audio_vae"), dict(
        base_channels = 4, ch_mult = [1], num_res_blocks = 1, latent_channels = 2, mel_bins = 8, resolution = 32))
    build_df_cls(LTX2VocoderWithBWE, j("vocoder"), o("vocoder"), dict(
        in_channels = av.config.output_channels * av.config.mel_bins, hidden_channels = 256, bwe_hidden_channels = 128,
        num_mel_channels = 8, bwe_in_channels = 2 * 8))
    build_df_cls(LTX2LatentUpsamplerModel, j("latent_upsampler"), o("latent_upsampler"), dict(
        in_channels = lat, mid_channels = 32, num_blocks_per_stage = 1))
    build_df_cls(LTX2DurationHead, j("duration_head"), o("duration_head"), dict(
        video_cross_attention_dim = 32, audio_cross_attention_dim = 32, pooler_hidden_dim = 8, num_pooler_heads = 2,
        mlp_hidden_dim = 8))
    # Studio never loads these (video_ltx25._PACK_SKIP_PREFIXES / LTX25_NONE_COMPONENTS) and a hub load never fetches
    # them; a LOCAL folder must still carry every indexed component, so the tiny index nulls the three it lacks.
    p = os.path.join(out, "model_index.json")
    index = json.load(open(p))
    for k in ("diffusion_decoder", "processor", "prompt_enhancer"):
        if k in index:
            index[k] = [None, None]
    json.dump(index, open(p, "w"), indent = 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--cfg", default = None)
    ap.add_argument("--backend", default = None, help = "studio/backend (MOSS in-tree transformer class)")
    ap.add_argument("fams", nargs = "+")
    a = ap.parse_args()
    cfg_root = a.cfg or os.path.join(a.out, "_cfg")
    for fam in a.fams:
        src = fetch_cfg(REPOS[fam], cfg_root)
        out = os.path.join(a.out, fam)
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        print("==", fam, REPOS[fam], "->", out, flush = True)
        if fam == "krea-2":
            krea2(src, out)
        elif fam == "flux.3-klein":
            flux3(src, out)
            fix_index(out)
        elif fam == "moss-image":
            moss(src, out, a.backend)
        elif fam == "ltx-2.5":
            ltx25(src, out)
        json.dump({"repo": REPOS[fam], "family": fam}, open(os.path.join(out, "_tiny_source.json"), "w"))
        print("OK", fam, flush = True)


if __name__ == "__main__":
    main()
