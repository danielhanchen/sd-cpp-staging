// conv microbench: CONV_3D direct (Wan VAE shapes) and CONV_2D direct (FLUX.2 VAE shapes) on the CPU backend.
// usage: conv_bench <threads> <reps> <case> [out.bin]
//   cases: w96 (3x3x3, 96->96, in 482x322x6), w192 (242x162x6, 192->192), w384 (122x82x4, 384->384),
//          c128 (3x3, 128->128, 1026x1026), c512 (3x3, 512->512, 258x258)
// prints per-rep ms; writes the f32 output for bit comparison.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint32_t rng = 12345;
static float frand() { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) & 0xffff) / 65536.0f - 0.5f; }

int main(int argc, char ** argv) {
    int nt = atoi(argv[1]), reps = atoi(argv[2]); std::string c = argv[3];
    int W, H, D, IC, OC; bool is3d = c[0] == 'w';
    if (c == "w96") { W = 482; H = 322; D = 6; IC = OC = 96; }
    else if (c == "w192") { W = 242; H = 162; D = 6; IC = OC = 192; }
    else if (c == "w384") { W = 122; H = 82; D = 4; IC = OC = 384; }
    else if (c == "c128") { W = 1026; H = 1026; D = 1; IC = OC = 128; }
    else if (c == "c512") { W = 258; H = 258; D = 1; IC = OC = 512; }
    else { fprintf(stderr, "bad case\n"); return 1; }
    ggml_init_params ip = { (size_t) 1 << 20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor *k, *x, *y;
    if (is3d) {
        k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 3, (int64_t) IC * OC);
        x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, D, IC);
        y = ggml_conv_3d_direct(ctx, k, x, 1, 1, 1, 0, 0, 0, 1, 1, 1, IC, 1, OC);
    } else {
        k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, IC, OC);
        x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, IC, 1);
        y = ggml_conv_2d_direct(ctx, k, x, 1, 1, 0, 0, 1, 1);
    }
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, nt);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<ggml_fp16_t> kh(ggml_nelements(k)); for (auto & v : kh) v = ggml_fp32_to_fp16(frand() * 0.1f);
    std::vector<float> xh(ggml_nelements(x)); for (auto & v : xh) v = frand();
    ggml_backend_tensor_set(k, kh.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(x, xh.data(), 0, ggml_nbytes(x));
    double flops = 2.0 * ggml_nelements(y) * (double) IC * (is3d ? 27 : 9);
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be, g);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, ms);
        printf("rep %d %.2f ms %.1f GFLOPS\n", r, ms, flops / ms / 1e6);
    }
    printf("case %s best %.2f ms %.1f GFLOPS\n", c.c_str(), best, flops / best / 1e6);
    if (argc > 4) {
        std::vector<float> out(ggml_nelements(y)); ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
        FILE * f = fopen(argv[4], "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f);
    }
    ggml_backend_buffer_free(buf); ggml_backend_free(be); ggml_free(ctx);
    return 0;
}
