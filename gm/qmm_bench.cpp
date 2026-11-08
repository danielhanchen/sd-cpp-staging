// qmm_bench: per weight type, time one ggml CPU MUL_MAT at diffusion shapes and check it against
// (a) the CPU backend reference path (use_ref: vec_dot, no llamafile / iqk / tiled) and (b) an fp64 GEMM
// over the dequantized weights. The weight sits either in the plain CPU buffer or in the first CPU extra
// buffer type (repack / AMX / KleidiAI) that accepts the op, the way stable-diffusion.cpp places matmul
// weights. Prints one JSON object per (type, shape, buffer) line.
//
// usage: qmm_bench [-t threads] [-r reps] [-s m,k,n]... [-T type,type,...] [-b plain|extra|both] [--oracle-cols N]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct shape { int64_t m, k, n; };

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static float frand() {  // deterministic, roughly normal (sum of 4 uniforms)
    float s = 0;
    for (int i = 0; i < 4; ++i) {
        g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
        s += (float) ((g_rng >> 11) * (1.0 / 9007199254740992.0)) - 0.5f;
    }
    return s * 1.7320508f;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

typedef void (*set_use_ref_t)(ggml_backend_t, bool);
typedef int64_t (*iqk_count_t)(int);

int main(int argc, char ** argv) {
    int threads = 4, reps = 3, oracle_cols = 32;
    std::vector<shape> shapes;
    std::vector<std::string> types;
    std::string bufs = "both";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-t") threads = atoi(next().c_str());
        else if (a == "-r") reps = atoi(next().c_str());
        else if (a == "-b") bufs = next();
        else if (a == "--oracle-cols") oracle_cols = atoi(next().c_str());
        else if (a == "-s") { shape s; if (sscanf(next().c_str(), "%lld,%lld,%lld", (long long *) &s.m, (long long *) &s.k, (long long *) &s.n) == 3) shapes.push_back(s); }
        else if (a == "-T") { std::string l = next(); size_t p = 0; while (p <= l.size()) { size_t q = l.find(',', p); if (q == std::string::npos) q = l.size(); if (q > p) types.push_back(l.substr(p, q - p)); p = q + 1; } }
    }
    if (shapes.empty()) shapes.push_back({3072, 3072, 1024});
    if (types.empty()) types = {"f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
                                "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs"};

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) { fprintf(stderr, "no CPU device\n"); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    auto set_use_ref = (set_use_ref_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    auto iqk_count = (iqk_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_iqk_count");
    if (set_threads) set_threads(be, threads);
    fprintf(stderr, "qmm_bench: device %s, threads %d, use_ref %s\n", ggml_backend_dev_description(dev), threads, set_use_ref ? "yes" : "no");

    for (const shape & sh : shapes) {
        const int64_t m = sh.m, k = sh.k, n = sh.n;
        std::vector<float> wf(m * k), xf(k * n);
        g_rng = 0x9E3779B97F4A7C15ull ^ (uint64_t) (m * 131 + k * 7 + n);
        for (auto & v : wf) v = frand() * 0.05f;
        for (auto & v : xf) v = frand();
        std::vector<float> imat(k, 1.0f);

        for (const std::string & tn : types) {
            ggml_type type = GGML_TYPE_COUNT;
            for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                const char * nm = ggml_type_name((ggml_type) t);
                if (nm && tn == nm) { type = (ggml_type) t; break; }
            }
            if (type == GGML_TYPE_COUNT || k % ggml_blck_size(type) != 0) { fprintf(stderr, "skip %s\n", tn.c_str()); continue; }
            const size_t qsize = ggml_row_size(type, k) * m;
            std::vector<uint8_t> q(qsize);
            ggml_quantize_chunk(type, wf.data(), q.data(), 0, m, k, ggml_quantize_requires_imatrix(type) ? imat.data() : nullptr);

            // fp64 oracle on the dequantized weights, first oracle_cols activation columns
            const int oc = (int) std::min<int64_t>(oracle_cols, n);
            std::vector<double> yo((size_t) m * oc);
            {
                std::vector<float> wd(m * k);
                const auto * tt = ggml_get_type_traits(type);
                if (type == GGML_TYPE_F32) wd = wf; else tt->to_float(q.data(), wd.data(), m * k);
                for (int64_t j = 0; j < oc; ++j)
                    for (int64_t i = 0; i < m; ++i) {
                        double s = 0; const float * a = &wd[i * k]; const float * b = &xf[j * k];
                        for (int64_t l = 0; l < k; ++l) s += (double) a[l] * b[l];
                        yo[j * m + i] = s;
                    }
            }

            std::vector<std::pair<std::string, ggml_backend_buffer_type_t>> cands;
            if (bufs != "extra") cands.push_back({"plain", ggml_backend_dev_buffer_type(dev)});
            std::vector<float> yref;
            double t_ref = -1;
            for (int pass = 0; pass < 2; ++pass) {
                // pass 0: reference (use_ref, plain buffer); pass 1: the candidate buffers with every fast path on
                std::vector<std::pair<std::string, ggml_backend_buffer_type_t>> list;
                if (pass == 0) list.push_back({"ref", ggml_backend_dev_buffer_type(dev)});
                else {
                    list = cands;
                    if (bufs != "plain" && get_extra) {
                        for (ggml_backend_buffer_type_t * it = get_extra(dev); it && *it; ++it) list.push_back({ggml_backend_buft_name(*it), *it});
                    }
                }
                for (auto & c : list) {
                    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
                    ggml_context * cw = ggml_init(ip);
                    ggml_context * cx = ggml_init(ip);
                    ggml_tensor * w = ggml_new_tensor_2d(cw, type, k, m);
                    ggml_tensor * x = ggml_new_tensor_2d(cx, GGML_TYPE_F32, k, n);
                    ggml_tensor * y = ggml_mul_mat(cx, w, x);
                    if (pass == 1 && c.first != "plain") {
                        // only an extra buffer type that takes this op, probed like sd_cpu_repack_buffer_type_for
                        ggml_backend_buffer_t probe = ggml_backend_buft_alloc_buffer(c.second, 0);
                        bool ok = false;
                        if (probe) { w->buffer = probe; ok = ggml_backend_dev_supports_op(dev, y); w->buffer = nullptr; ggml_backend_buffer_free(probe); }
                        if (!ok) { ggml_free(cw); ggml_free(cx); continue; }
                    }
                    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, c.second);
                    ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors_from_buft(cx, ggml_backend_dev_buffer_type(dev));
                    if (!bw || !bx) { fprintf(stderr, "alloc failed %s %s\n", tn.c_str(), c.first.c_str()); continue; }
                    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    ggml_backend_tensor_set(w, q.data(), 0, qsize);
                    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
                    ggml_cgraph * gf = ggml_new_graph(cx);
                    ggml_build_forward_expand(gf, y);
                    if (set_use_ref) set_use_ref(be, pass == 0);
                    const int64_t iqk0 = iqk_count ? iqk_count(type) : 0;
                    std::vector<double> ts;
                    for (int r = 0; r < reps + 1; ++r) {
                        double t0 = now_ms();
                        ggml_backend_graph_compute(be, gf);
                        double t1 = now_ms();
                        if (r > 0) ts.push_back(t1 - t0);
                    }
                    const int64_t iqk1 = iqk_count ? iqk_count(type) : 0;
                    std::vector<float> yv(m * n);
                    ggml_backend_tensor_get(y, yv.data(), 0, yv.size() * sizeof(float));
                    std::sort(ts.begin(), ts.end());
                    const double med = ts[ts.size() / 2];
                    double se_o = 0, so = 0, se_r = 0, max_rel_r = 0; int64_t neq = 0;
                    for (int64_t j = 0; j < oc; ++j) for (int64_t i = 0; i < m; ++i) {
                        double d = yv[j * m + i] - yo[j * m + i]; se_o += d * d; so += yo[j * m + i] * yo[j * m + i];
                    }
                    if (pass == 0) { yref = yv; t_ref = med; }
                    double sr = 0;
                    for (int64_t z = 0; z < m * n; ++z) {
                        double d = (double) yv[z] - yref[z]; se_r += d * d; sr += (double) yref[z] * yref[z];
                        if (yv[z] != yref[z]) ++neq;
                        double rel = std::fabs(d) / (std::fabs((double) yref[z]) + 1e-3);
                        if (rel > max_rel_r) max_rel_r = rel;
                    }
                    uint64_t hash = 1469598103934665603ull;  // FNV-1a over the output bits
                    for (int64_t z = 0; z < m * n; ++z) {
                        uint32_t u; memcpy(&u, &yv[z], 4);
                        for (int b = 0; b < 4; ++b) { hash ^= (u >> (8 * b)) & 0xff; hash *= 1099511628211ull; }
                    }
                    const double gflops = 2.0 * m * k * n / (med * 1e6);
                    printf("{\"type\":\"%s\",\"m\":%lld,\"k\":%lld,\"n\":%lld,\"buf\":\"%s\",\"threads\":%d,\"ms_med\":%.3f,\"ms_min\":%.3f,"
                           "\"ms_max\":%.3f,\"gflops\":%.1f,\"speedup_vs_ref\":%.3f,\"iqk_calls\":%lld,\"nmse_oracle\":%.4e,\"nmse_vs_ref\":%.4e,"
                           "\"frac_neq_ref\":%.5f,\"max_rel_vs_ref\":%.3e,\"hash\":\"%016llx\"}\n",
                           tn.c_str(), (long long) m, (long long) k, (long long) n, c.first.c_str(), threads, med, ts.front(), ts.back(), gflops,
                           t_ref > 0 ? t_ref / med : 0.0, (long long) (iqk1 - iqk0), se_o / (so + 1e-300), se_r / (sr + 1e-300),
                           (double) neq / (m * n), max_rel_r, (unsigned long long) hash);
                    fflush(stdout);
                    ggml_backend_buffer_free(bw); ggml_backend_buffer_free(bx);
                    ggml_free(cw); ggml_free(cx);
                }
            }
        }
    }
    if (set_use_ref) set_use_ref(be, false);
    ggml_backend_free(be);
    return 0;
}
