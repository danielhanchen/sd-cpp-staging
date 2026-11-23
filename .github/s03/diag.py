#!/usr/bin/env python3
"""Harness-only edits to a ggml tree (both arms, inert unless the env var is set).

usage: diag.py GGML_DIR
- GGML_METAL_FORCE_SIMDGROUP=1: report simdgroup reduction + matrix mul on the paravirtual GPU (the host is an M1),
  so Metal runs the same kernels a real M1 does.
- S03_CONV_LOG=1: one stderr line per CONV_2D dispatch naming the path (mm = implicit GEMM, scalar = direct kernel).
- test-backend-ops perf: CONV_2D cases at the SD / FLUX VAE decoder shapes for a 1024 px image (pad 1, F16 kernel).
"""
import sys
from pathlib import Path

g = Path(sys.argv[1])


def sub(path, old, new, count=1):
    p = g / path
    s = p.read_text()
    n = s.count(old)
    assert n == count, (path, old[:60], n)
    p.write_text(s.replace(old, new))


sub("src/ggml-metal/ggml-metal-device.m",
    "                dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];\n",
    "                dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];\n"
    "                if (getenv(\"GGML_METAL_FORCE_SIMDGROUP\") != NULL) {\n"
    "                    dev->props.has_simdgroup_reduction = true;\n"
    "                    dev->props.has_simdgroup_mm        = true;\n"
    "                }\n")

ops = "src/ggml-metal/ggml-metal-ops.cpp"
sub(ops, "#include <cassert>\n", "#include <cassert>\n#include <cstdio>\n#include <cstdlib>\n")
s = (g / ops).read_text()
if "ggml_metal_library_get_pipeline_conv_2d_mm(lib, op);" in s:
    sub(ops, "        auto pipeline = ggml_metal_library_get_pipeline_conv_2d_mm(lib, op);\n",
        "        auto pipeline = ggml_metal_library_get_pipeline_conv_2d_mm(lib, op);\n"
        "        if (getenv(\"S03_CONV_LOG\")) { fprintf(stderr, \"S03_CONV2D mm\\n\"); }\n")
sub(ops, "    auto pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);\n",
    "    auto pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);\n"
    "    if (getenv(\"S03_CONV_LOG\")) { fprintf(stderr, \"S03_CONV2D scalar\\n\"); }\n")

tbo = "tests/test-backend-ops.cpp"
anchor = "static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {\n    std::vector<std::unique_ptr<test_case>> test_cases;\n"
sub(tbo, anchor, anchor + """
    // s03 harness: SD / FLUX VAE decoder CONV_2D at a 1024 px output, {W, Cin, Cout}, 3x3, pad 1
    for (auto c : std::vector<std::array<int, 3>>{
             {128, 512, 512}, {256, 512, 512}, {512, 512, 256}, {512, 256, 256}, {1024, 256, 128}, {1024, 128, 128}, {1024, 128, 3}}) {
        test_cases.emplace_back(new test_conv_2d({ c[0], c[0], c[1], 1 }, { 3, 3, c[1], c[2] }, GGML_TYPE_F16, 1, 1, 1, 1, 1, 1, false));
    }
""")
print("diag ok:", g)
