// qtest <base libggml-cpu-X.so> <patched libggml-cpu-X.so>: byte-compare ggml_quantize_mat_q8_K_4x{4,8} between
// the base build (generic C under the public name) and the patched build (PR #32554 NEON under the public name,
// generic under *_generic when exported). Inputs: random normal rows, scaled rows, ties at max (+/-), zeros.
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <string>
typedef void (*qfn)(const float *, void *, int64_t);
static const int QK = 256;
static const size_t BLK = 4 * 4 + QK * 4 + (QK / 4) * 2;

int main(int argc, char ** argv) {
    void * hb = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    void * hp = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!hb || !hp) { printf("dlopen fail %s\n", dlerror()); return 2; }
    int bad_total = 0;
    for (const char * nm : {"ggml_quantize_mat_q8_K_4x4", "ggml_quantize_mat_q8_K_4x8"}) {
        qfn fb = (qfn) dlsym(hb, nm), fp = (qfn) dlsym(hp, nm);
        qfn fg = (qfn) dlsym(hp, (std::string(nm) + "_generic").c_str());
        printf("%s base=%p pat=%p pat_generic=%p\n", nm, (void *) fb, (void *) fp, (void *) fg);
        if (!fb || !fp) { bad_total++; continue; }
        std::mt19937 rng(1);
        std::normal_distribution<float> nd(0.f, 1.f);
        const int K = 3072;
        int nbad = 0, ncase = 0; long first_off = -1; int first_case = -1;
        for (int c = 0; c < 400; c++) {
            std::vector<float> x(4 * K);
            float sc = std::pow(10.f, (c % 9) - 4.f);
            for (auto & v : x) v = nd(rng) * sc;
            if (c % 5 == 1) for (int r = 0; r < 4; r++) { x[r * K + 7] = 3 * sc; x[r * K + 100] = -3 * sc; }   // tie +/- at max
            if (c % 5 == 2) for (int r = 0; r < 4; r++) { x[r * K + 7] = -3 * sc; x[r * K + 100] = 3 * sc; }
            if (c % 5 == 3) for (int i = 0; i < K; i++) x[i] = 0.f;                                              // zero row
            if (c % 5 == 4) for (auto & v : x) v = std::round(v * 4) / 4;                                        // many exact halves
            std::vector<uint8_t> ob((K / QK) * BLK, 0xAA), op((K / QK) * BLK, 0x55), og((K / QK) * BLK, 0x33);
            fb(x.data(), ob.data(), K); fp(x.data(), op.data(), K);
            ncase++;
            if (memcmp(ob.data(), op.data(), ob.size())) {
                nbad++;
                if (first_off < 0) { for (size_t i = 0; i < ob.size(); i++) if (ob[i] != op[i]) { first_off = (long) i; break; } first_case = c; }
            }
            if (fg) { fg(x.data(), og.data(), K); if (memcmp(ob.data(), og.data(), ob.size())) printf("  generic-in-pat differs from base, case %d\n", c); }
        }
        printf("RESULT %s cases=%d mismatched=%d first_case=%d first_byte_off_in_row_block=%ld (block bytes %zu)\n", nm, ncase, nbad,
               first_case, first_off, BLK);
        bad_total += nbad;
    }
    return bad_total ? 1 : 0;
}
