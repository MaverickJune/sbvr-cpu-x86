// sbvr_kernel_x86.cpp ─────────────────────────────────────────────────────────
// x86-64 optimised SBVR GEMV kernel  (AVX2 + FMA + F16C baseline,
//                                      optional AVX-512 BITALG fast-path)
//
// This is a port of the ARM-NEON sbvr_kernel.cpp.
// Major architectural differences vs. the NEON original:
//   • N_LANE = 32  (256-bit __m256i holds 32 × uint8)  vs. 16 on NEON
//   • Byte popcount: LUT-shuffle on AVX2, native _mm256_popcnt_epi8 on
//     AVX-512 BITALG+VL (Ice Lake / Zen 4+)
//   • No native __fp16 arithmetic – all math is float32; F16C converts
//     at load/store boundaries
//   • FMA3 _mm256_fmadd_ps replaces NEON vfmlalq_*_f16
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/extension.h>
#include <iostream>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <vector>
#include <algorithm>
#include <immintrin.h>          // AVX2 / FMA / F16C / AVX-512

#include "thread_pool.hpp"

// ─── constants ───────────────────────────────────────────────────────────────
#define K_PER_BVR  32           // 256-bit BVR → 32 bytes
#define N_LANE     32           // AVX2: __m256i = 32 × uint8 lanes

// ─── thread-pool singleton (unchanged from NEON version) ─────────────────────
static ThreadPool& global_pool()
{
    static ThreadPool pool;
    return pool;
}

extern "C" void sbvr_init_pool(int num_threads)
{
    global_pool().init(num_threads);
}

extern "C" void sbvr_finalize_pool()
{
    global_pool().finalize();
}

// ─── fp16 storage type ───────────────────────────────────────────────────────
// x86 has no hardware __fp16 type; at::Half is a 16-bit struct that is
// bit-identical to IEEE-754 binary16.  We pass pointers around as at::Half*
// and reinterpret to uint16_t* / __m128i* for SIMD load/convert.
using fp16_t = at::Half;

// ─── function-pointer type for the dispatch table ────────────────────────────
typedef void (*KernelLaunchFn)(
    uint8_t* l_bvr,  void* l_coeff_idx,  fp16_t* l_coeff_cache,
    uint8_t* r_bvr,  void* r_coeff_idx,  fp16_t* r_coeff_cache,
    fp16_t*  bias,   fp16_t* out,
    int M, int N, int K);

// ─── small coefficient packet (same idea as NEON version) ────────────────────
template <int NUM_SUMS>
struct coeffs {
    fp16_t i[NUM_SUMS];
};

// ─── per-task descriptor for the thread pool ─────────────────────────────────
template<typename LIndexT, typename RIndexT,
         int LNumSums, int RNumSums>
struct WorkerArg {
    const uint8_t*  l_bvr;
    const LIndexT*  l_coeff_idx;
    const fp16_t*   l_coeff_cache;

    const uint8_t*  r_bvr;
    const RIndexT*  r_coeff_idx;
    const fp16_t*   r_coeff_cache;

    fp16_t*         bias_pack;          // may be nullptr
    fp16_t*         out_pack;

    int N;                              // columns handled by this task
    int K;
};


// ═════════════════════════════════════════════════════════════════════════════
//  SIMD helpers
// ═════════════════════════════════════════════════════════════════════════════

// ─── per-byte popcount (32 lanes) ────────────────────────────────────────────
//  • AVX-512 BITALG+VL  → single instruction
//  • AVX2 fallback       → classic nibble-LUT via vpshufb
#ifdef __AVX512BITALG__

static inline __m256i byte_popcount(__m256i v)
{
    return _mm256_popcnt_epi8(v);       // 1 µop, 1-cycle latency (ICL/Zen4)
}

#else  // AVX2 fallback

static inline __m256i byte_popcount(__m256i v)
{
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2, 1,2,2,3, 1,2,2,3, 2,3,3,4,
        0,1,1,2, 1,2,2,3, 1,2,2,3, 2,3,3,4);

    const __m256i lo_mask = _mm256_set1_epi8(0x0F);

    __m256i lo = _mm256_and_si256(v, lo_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask);

    return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                           _mm256_shuffle_epi8(lut, hi));
}

#endif // __AVX512BITALG__


// ─── fp16 ↔ fp32 bulk helpers (F16C) ─────────────────────────────────────────
// Load 8 × fp16 → __m256 (8 × float32)
static inline __m256 load_f16x8_as_f32(const fp16_t* p)
{
    __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_cvtph_ps(bits);
}

// Store __m256 (8 × float32) → 8 × fp16
static inline void store_f32_as_f16x8(fp16_t* p, __m256 v)
{
    __m128i bits = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), bits);
}

// Load single fp16 → float
static inline float load_f16_scalar(const fp16_t* p)
{
    uint16_t raw;
    std::memcpy(&raw, p, 2);
    __m128i v = _mm_cvtsi32_si128(raw);
    __m128  f = _mm_cvtph_ps(v);            // F16C scalar convert
    return _mm_cvtss_f32(f);
}


// ═════════════════════════════════════════════════════════════════════════════
//  Core micro-kernel:  1 × N_LANE  (M=1, 32 output columns per tile)
//
//  Algorithm (popcount-first strategy, matching ARM popc_first variant):
//    For each BVR block (256-bit / 32-byte granularity):
//      1.  Accumulate AND-popcount across K_PER_BVR=32 k-steps into
//          popc_cache[LNumSums][RNumSums]  (__m256i, 32 × uint8).
//      2.  Gather right-side coefficients into a float lane_tile so that
//          the multiply phase is a pure SIMD loop.
//      3.  For every (l, r) pair, widen popc → float32, then
//              acc += (l_coeff * r_coeff) * popcount
//          using FMA3.
//    Finally convert the 32 float32 accumulators to fp16, add bias, store.
// ═════════════════════════════════════════════════════════════════════════════

template<
    typename  LIndexT,  typename  RIndexT,
    int       LNumSums, int       RNumSums>
static inline void
simd_kernel_1xN_x86(
    const uint8_t*  __restrict l_bvr,           // flat (K, LNumSums)
    const LIndexT*  __restrict l_coeff_idx,     // (K/K_PER_BVR,)
    const fp16_t*   __restrict l_coeff_cache,   // (cache_size, LNumSums)

    const uint8_t*  __restrict r_bvr,           // (N/N_LANE, K, RNumSums, N_LANE=32)
    const RIndexT*  __restrict r_coeff_idx,     // (N, K/K_PER_BVR)
    const fp16_t*   __restrict r_coeff_cache,   // (cache_size, RNumSums)

    const fp16_t*   __restrict bias_pack,       // (N,) or nullptr
    fp16_t*         __restrict out_pack,         // (N,)
    int N, int K)
{
    const int bvr_per_K = K / K_PER_BVR;

    for (int n = 0; n < N; n += N_LANE)
    {
        // ── 32 float32 accumulators (4 × __m256, lanes 0-7 / 8-15 / 16-23 / 24-31)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        // ── scratch: right-side coefficients gathered per-lane (already float32)
        alignas(32) float lane_tile[RNumSums][N_LANE];

        for (int bvr_idx = 0; bvr_idx < bvr_per_K; ++bvr_idx)
        {
            // ────────────────────────────────────────────────────────────────
            //  Phase 0 – gather right coefficients for every lane
            // ────────────────────────────────────────────────────────────────
            for (int nn = 0; nn < N_LANE; ++nn)
            {
                const int r_ci = r_coeff_idx[(n + nn) * bvr_per_K + bvr_idx];
                const fp16_t* src = r_coeff_cache + r_ci * RNumSums;
                for (int r = 0; r < RNumSums; ++r)
                    lane_tile[r][nn] = load_f16_scalar(&src[r]);
            }

            // ────────────────────────────────────────────────────────────────
            //  Phase 1 – AND + byte-popcount accumulation
            // ────────────────────────────────────────────────────────────────
            alignas(32) __m256i popc_cache[LNumSums][RNumSums];
            for (int l = 0; l < LNumSums; ++l)
                for (int r = 0; r < RNumSums; ++r)
                    popc_cache[l][r] = _mm256_setzero_si256();

            for (int k = 0; k < K_PER_BVR; ++k)
            {
                const int k_idx = bvr_idx * K_PER_BVR + k;

                for (int l_idx = 0; l_idx < LNumSums / 2; ++l_idx)
                {
                    // Left: single byte broadcast to all 32 lanes
                    const uint8_t l0_b = l_bvr[k_idx * LNumSums + l_idx * 2];
                    const uint8_t l1_b = l_bvr[k_idx * LNumSums + l_idx * 2 + 1];

                    const __m256i l0 = _mm256_set1_epi8(static_cast<char>(l0_b));
                    const __m256i l1 = _mm256_set1_epi8(static_cast<char>(l1_b));

                    for (int r_idx = 0; r_idx < RNumSums / 2; ++r_idx)
                    {
                        // Right: load 32 bytes (one per output lane) for each of two sum indices
                        const uint8_t* r_base =
                            &r_bvr[n * K * RNumSums
                                   + (k_idx * RNumSums + r_idx * 2) * N_LANE];

                        const __m256i r0 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(r_base));
                        const __m256i r1 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(r_base + N_LANE));

                        const int li0 = l_idx * 2;
                        const int li1 = li0 + 1;
                        const int rj0 = r_idx * 2;
                        const int rj1 = rj0 + 1;

                        popc_cache[li0][rj0] = _mm256_add_epi8(
                            popc_cache[li0][rj0],
                            byte_popcount(_mm256_and_si256(l0, r0)));

                        popc_cache[li0][rj1] = _mm256_add_epi8(
                            popc_cache[li0][rj1],
                            byte_popcount(_mm256_and_si256(l0, r1)));

                        popc_cache[li1][rj0] = _mm256_add_epi8(
                            popc_cache[li1][rj0],
                            byte_popcount(_mm256_and_si256(l1, r0)));

                        popc_cache[li1][rj1] = _mm256_add_epi8(
                            popc_cache[li1][rj1],
                            byte_popcount(_mm256_and_si256(l1, r1)));
                    }
                }
            }

            // ────────────────────────────────────────────────────────────────
            //  Phase 2 – coefficient × popcount → accumulate into float32
            // ────────────────────────────────────────────────────────────────
            const int l_ci = l_coeff_idx[bvr_idx];

            for (int l = 0; l < LNumSums; ++l)
            {
                const float a_val = load_f16_scalar(
                    &l_coeff_cache[l_ci * LNumSums + l]);
                const __m256 a_vec = _mm256_set1_ps(a_val);

                for (int r = 0; r < RNumSums; ++r)
                {
                    // Load pre-gathered right coefficients (already float32)
                    const __m256 b0 = _mm256_load_ps(&lane_tile[r][ 0]);
                    const __m256 b1 = _mm256_load_ps(&lane_tile[r][ 8]);
                    const __m256 b2 = _mm256_load_ps(&lane_tile[r][16]);
                    const __m256 b3 = _mm256_load_ps(&lane_tile[r][24]);

                    // Widen uint8 popcounts → float32
                    //   __m256i (32 × u8)  →  split into 2 × __m128i
                    //                      →  cvtepu8_epi16 (16 × u16 each)
                    //                      →  cvtepu16_epi32 (8 × u32 each)
                    //                      →  cvtepi32_ps    (8 × f32 each)
                    const __m256i pc = popc_cache[l][r];
                    const __m128i pc_lo128 = _mm256_castsi256_si128(pc);
                    const __m128i pc_hi128 = _mm256_extracti128_si256(pc, 1);

                    // Lanes 0-15 (lower 128 bits)
                    const __m256i pc16_a = _mm256_cvtepu8_epi16(pc_lo128);
                    const __m256  pcf0   = _mm256_cvtepi32_ps(
                        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(pc16_a)));
                    const __m256  pcf1   = _mm256_cvtepi32_ps(
                        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(pc16_a, 1)));

                    // Lanes 16-31 (upper 128 bits)
                    const __m256i pc16_b = _mm256_cvtepu8_epi16(pc_hi128);
                    const __m256  pcf2   = _mm256_cvtepi32_ps(
                        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(pc16_b)));
                    const __m256  pcf3   = _mm256_cvtepi32_ps(
                        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(pc16_b, 1)));

                    // acc += (a * b) * popcount       [FMA3]
                    const __m256 ab0 = _mm256_mul_ps(a_vec, b0);
                    const __m256 ab1 = _mm256_mul_ps(a_vec, b1);
                    const __m256 ab2 = _mm256_mul_ps(a_vec, b2);
                    const __m256 ab3 = _mm256_mul_ps(a_vec, b3);

                    acc0 = _mm256_fmadd_ps(ab0, pcf0, acc0);
                    acc1 = _mm256_fmadd_ps(ab1, pcf1, acc1);
                    acc2 = _mm256_fmadd_ps(ab2, pcf2, acc2);
                    acc3 = _mm256_fmadd_ps(ab3, pcf3, acc3);
                }
            }

        } // bvr_idx

        // ── add bias (if present) ────────────────────────────────────────────
        if (bias_pack)
        {
            acc0 = _mm256_add_ps(acc0, load_f16x8_as_f32(bias_pack + n));
            acc1 = _mm256_add_ps(acc1, load_f16x8_as_f32(bias_pack + n + 8));
            acc2 = _mm256_add_ps(acc2, load_f16x8_as_f32(bias_pack + n + 16));
            acc3 = _mm256_add_ps(acc3, load_f16x8_as_f32(bias_pack + n + 24));
        }

        // ── float32 → fp16 store ────────────────────────────────────────────
        store_f32_as_f16x8(out_pack + n,      acc0);
        store_f32_as_f16x8(out_pack + n +  8, acc1);
        store_f32_as_f16x8(out_pack + n + 16, acc2);
        store_f32_as_f16x8(out_pack + n + 24, acc3);

    } // n
}


// ═════════════════════════════════════════════════════════════════════════════
//  Scalar reference kernel (simd_kernel_ver2 equivalent, for validation)
// ═════════════════════════════════════════════════════════════════════════════

template<
    typename  LIndexT,  typename  RIndexT,
    int       LNumSums, int       RNumSums>
static inline void
scalar_kernel_x86(
    const uint8_t*  __restrict l_bvr,
    const LIndexT*  __restrict l_coeff_idx,
    const fp16_t*   __restrict l_coeff_cache,

    const uint8_t*  __restrict r_bvr,
    const RIndexT*  __restrict r_coeff_idx,
    const fp16_t*   __restrict r_coeff_cache,

    const fp16_t*   __restrict bias_pack,
    fp16_t*         __restrict out_pack,
    int N, int K)
{
    const int bvr_per_K = K / K_PER_BVR;

    for (int nn = 0; nn < N; ++nn)
    {
        float acc = 0.f;

        for (int bvr_idx = 0; bvr_idx < bvr_per_K; ++bvr_idx)
        {
            const int l_ci = l_coeff_idx[bvr_idx];
            const int r_ci = r_coeff_idx[nn * bvr_per_K + bvr_idx];

            for (int l = 0; l < LNumSums; ++l)
            {
                const float a = load_f16_scalar(&l_coeff_cache[l_ci * LNumSums + l]);

                for (int r = 0; r < RNumSums; ++r)
                {
                    const float b = load_f16_scalar(&r_coeff_cache[r_ci * RNumSums + r]);

                    // popcount(AND) across K_PER_BVR bytes
                    int popc = 0;
                    for (int k = 0; k < K_PER_BVR; ++k)
                    {
                        const int k_idx = bvr_idx * K_PER_BVR + k;
                        uint8_t lv = l_bvr[k_idx * LNumSums + l];

                        // r_bvr layout: (N/N_LANE, K, RNumSums, N_LANE)
                        // nn = local output column within this task's sub-range
                        const int n_group = nn / N_LANE;
                        const int n_lane  = nn % N_LANE;
                        uint8_t rv = r_bvr[n_group * K * RNumSums * N_LANE
                                           + k_idx * RNumSums * N_LANE
                                           + r * N_LANE
                                           + n_lane];
                        popc += __builtin_popcount(lv & rv);
                    }

                    acc += a * b * static_cast<float>(popc);
                }
            }
        }

        float bias_val = bias_pack
            ? load_f16_scalar(&bias_pack[nn])
            : 0.f;

        // store as fp16
        float result = acc + bias_val;
        __m128i h = _mm_cvtps_ph(_mm_set_ss(result),
                                 _MM_FROUND_TO_NEAREST_INT);
        uint16_t raw = static_cast<uint16_t>(_mm_cvtsi128_si32(h));
        std::memcpy(&out_pack[nn], &raw, 2);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
//  Thread-parallel dispatch  (same structure as NEON version)
// ═════════════════════════════════════════════════════════════════════════════

template<
    typename  LIndexT,  typename  RIndexT,
    int       LNumSums, int       RNumSums>
void sbvr_mm_cpu_1xN(
    uint8_t* l_bvr,  void* l_coeff_idx,  fp16_t* l_coeff_cache,
    uint8_t* r_bvr,  void* r_coeff_idx,  fp16_t* r_coeff_cache,
    fp16_t*  bias,   fp16_t* out,
    int M, int N, int K)
{
    const int num_threads = global_pool().num_threads();
    const int chunk_raw   = (N + num_threads - 1) / num_threads;
    // Round chunk UP to a multiple of N_LANE so every task is lane-aligned
    const int chunk       = ((chunk_raw + N_LANE - 1) / N_LANE) * N_LANE;

    const int bvr_per_K   = K / K_PER_BVR;

    std::vector<WorkerArg<LIndexT, RIndexT, LNumSums, RNumSums>> args;
    int n_tasks = 0;

    for (int t = 0; t < num_threads; ++t)
    {
        int n0 = t * chunk;
        int n1 = std::min(n0 + chunk, N);
        if (n0 >= n1) break;

        int n_items = n1 - n0;
        if (n_items % N_LANE != 0) {
            std::cerr << "Error: N-slice (" << n_items
                      << ") is not a multiple of N_LANE=" << N_LANE << "\n";
            throw std::runtime_error("Invalid N value for x86 kernel");
        }

        args.push_back({
            l_bvr,
            reinterpret_cast<const LIndexT*>(l_coeff_idx),
            l_coeff_cache,

            r_bvr  + n0 * K * RNumSums,                 // r_pack
            reinterpret_cast<const RIndexT*>(r_coeff_idx)
                + n0 * bvr_per_K,                        // r_coeff_idx_pack
            r_coeff_cache,

            bias ? bias + n0 : nullptr,                  // bias_pack
            out + n0,                                    // out_pack

            n_items,
            K
        });
        ++n_tasks;
    }

    global_pool().parallel_for(n_tasks, [&](int task_id) {
        const auto& a = args[task_id];
        simd_kernel_1xN_x86<LIndexT, RIndexT, LNumSums, RNumSums>(
            a.l_bvr, a.l_coeff_idx, a.l_coeff_cache,
            a.r_bvr, a.r_coeff_idx, a.r_coeff_cache,
            a.bias_pack, a.out_pack,
            a.N, a.K);
    });
}


// ═════════════════════════════════════════════════════════════════════════════
//  Template dispatch (LNumSums × RNumSums ∈ {2,4,6,8,10}²)
// ═════════════════════════════════════════════════════════════════════════════

template <typename LIndexT, typename RIndexT>
void launch_cpu_sbvr_kernel_wrapper(
    uint8_t* l_bvr,  void* l_coeff_idx,  fp16_t* l_coeff_cache,
    uint8_t* r_bvr,  void* r_coeff_idx,  fp16_t* r_coeff_cache,
    fp16_t*  bias,   fp16_t* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size)
{
    // 5×5 table:  LNumSums ∈ {2,4,6,8,10},  RNumSums ∈ {2,4,6,8,10}
    KernelLaunchFn kernel_list[] = {
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  2,  2>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  2,  4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  2,  6>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  2,  8>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  2, 10>,

        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  4,  2>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  4,  4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  4,  6>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  4,  8>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  4, 10>,

        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  6,  2>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  6,  4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  6,  6>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  6,  8>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  6, 10>,

        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  8,  2>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  8,  4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  8,  6>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  8,  8>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT,  8, 10>,

        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 10,  2>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 10,  4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 10,  6>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 10,  8>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 10, 10>,
    };

    const int idx = (l_num_sums - 2) / 2 * 5 + (r_num_sums - 2) / 2;
    if (idx < 0 || idx >= 25) {
        std::cerr << "Invalid kernel index: " << idx << "\n";
        throw std::runtime_error("Invalid kernel index");
    }

    kernel_list[idx](
        l_bvr, l_coeff_idx, l_coeff_cache,
        r_bvr, r_coeff_idx, r_coeff_cache,
        bias,  out,
        M, N, K);
}


// ═════════════════════════════════════════════════════════════════════════════
//  Top-level C++ dispatcher  (index-type routing)
// ═════════════════════════════════════════════════════════════════════════════

void launch_cpu_sbvr_mm_1xN(
    uint8_t* l_bvr,  void* l_coeff_idx,  fp16_t* l_coeff_cache,
    uint8_t* r_bvr,  void* r_coeff_idx,  fp16_t* r_coeff_cache,
    fp16_t*  bias,   fp16_t* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size)
{
    const bool use_l_u8 = (l_cache_size <= 256);
    const bool use_r_u8 = (r_cache_size <= 256);

    const bool ok =
        (l_num_sums % 2 == 0 && r_num_sums % 2 == 0) &&
        (l_num_sums >= 2 && l_num_sums <= 10) &&
        (r_num_sums >= 2 && r_num_sums <= 10);

    if (!ok) {
        std::cerr << "Unsupported config: l_num_sums=" << l_num_sums
                  << " r_num_sums=" << r_num_sums
                  << " l_cache=" << l_cache_size
                  << " r_cache=" << r_cache_size << "\n";
        return;
    }

    if (use_l_u8 && use_r_u8)
        launch_cpu_sbvr_kernel_wrapper<uint8_t, uint8_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out, M, N, K,
            l_num_sums, r_num_sums, l_cache_size, r_cache_size);
    else if (use_l_u8 && !use_r_u8)
        launch_cpu_sbvr_kernel_wrapper<uint8_t, uint16_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out, M, N, K,
            l_num_sums, r_num_sums, l_cache_size, r_cache_size);
    else if (!use_l_u8 && use_r_u8)
        launch_cpu_sbvr_kernel_wrapper<uint16_t, uint8_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out, M, N, K,
            l_num_sums, r_num_sums, l_cache_size, r_cache_size);
    else
        launch_cpu_sbvr_kernel_wrapper<uint16_t, uint16_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out, M, N, K,
            l_num_sums, r_num_sums, l_cache_size, r_cache_size);
}


// ═════════════════════════════════════════════════════════════════════════════
//  PyTorch entry-point
// ═════════════════════════════════════════════════════════════════════════════

torch::Tensor sbvr_cpu_mm_T(
    torch::Tensor l_bvr,            // (K, M=1, LNumSums)       uint8
    torch::Tensor l_coeff_idx,      // (K/K_PER_BVR, M)         uint8 or uint16
    torch::Tensor l_coeff_cache,    // (l_cache_size, LNumSums) fp16
    torch::Tensor r_bvr,            // (N/N_LANE, K, RNumSums, N_LANE=32) uint8
    torch::Tensor r_coeff_idx,      // (N, K/K_PER_BVR)         uint8 or uint16
    torch::Tensor r_coeff_cache,    // (r_cache_size, RNumSums) fp16
    torch::Tensor bias)
{
    const int M = l_bvr.size(1);
    const int N = r_bvr.size(0) * r_bvr.size(3);      // (N/N_LANE) * N_LANE
    const int K = l_bvr.size(0);
    const int l_num_sums  = l_bvr.size(2);
    const int r_num_sums  = r_bvr.size(2);
    const int l_cache_sz  = l_coeff_cache.size(0);
    const int r_cache_sz  = r_coeff_cache.size(0);

    TORCH_CHECK(r_bvr.size(3) == N_LANE,
        "r_bvr last dim must be N_LANE=", N_LANE,
        " for x86 kernel, got ", r_bvr.size(3));

    auto out = torch::empty({M, N},
        torch::dtype(torch::kFloat16).device(l_bvr.device()));

    fp16_t* bias_ptr = nullptr;
    if (bias.defined() && bias.numel() > 0 && bias.size(0) == N)
        bias_ptr = reinterpret_cast<fp16_t*>(bias.data_ptr<at::Half>());

    launch_cpu_sbvr_mm_1xN(
        l_bvr.data_ptr<uint8_t>(),
        l_coeff_idx.data_ptr(),
        reinterpret_cast<fp16_t*>(l_coeff_cache.data_ptr<at::Half>()),
        r_bvr.data_ptr<uint8_t>(),
        r_coeff_idx.data_ptr(),
        reinterpret_cast<fp16_t*>(r_coeff_cache.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<fp16_t*>(out.data_ptr<at::Half>()),
        M, N, K,
        l_num_sums, r_num_sums,
        l_cache_sz, r_cache_sz);

    return out;
}


// ═════════════════════════════════════════════════════════════════════════════
//  ISA capability check (informational)
// ═════════════════════════════════════════════════════════════════════════════
void sbvr_x86_test()
{
    // Quick AVX2 + F16C sanity check
    __m256 a = _mm256_setr_ps(1,2,3,4,5,6,7,8);
    __m256 b = _mm256_setr_ps(10,20,30,40,50,60,70,80);
    __m256 c = _mm256_add_ps(a, b);

    alignas(32) float out[8];
    _mm256_store_ps(out, c);
    std::cout << "AVX2 result: ";
    for (int i = 0; i < 8; ++i) std::cout << out[i] << " ";
    std::cout << "\n";

    // F16C round-trip
    __m128i h = _mm256_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);
    __m256  r = _mm256_cvtph_ps(h);
    _mm256_store_ps(out, r);
    std::cout << "F16C round-trip: ";
    for (int i = 0; i < 8; ++i) std::cout << out[i] << " ";
    std::cout << "\n";

#ifdef __AVX512BITALG__
    std::cout << "AVX-512 BITALG:  enabled  (native byte popcount)\n";
#else
    std::cout << "AVX-512 BITALG:  NOT available  (using LUT popcount)\n";
#endif

#ifdef __FMA__
    std::cout << "FMA3:            enabled\n";
#else
    std::cout << "FMA3:            NOT available\n";
#endif
}


// ═════════════════════════════════════════════════════════════════════════════
//  pybind11 module
// ═════════════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_sbvr_x86_test", &sbvr_x86_test,
          "AVX2 / F16C / FMA capability check");

    m.def("_sbvr_cpu_mm_T", &sbvr_cpu_mm_T,
          "SBVR GEMV on x86-64 (AVX2+FMA+F16C, optional AVX-512 BITALG)",
          py::arg("l_bvr"),
          py::arg("l_coeff_idx"),
          py::arg("l_coeff_cache"),
          py::arg("r_bvr"),
          py::arg("r_coeff_idx"),
          py::arg("r_coeff_cache"),
          py::arg("bias") = torch::Tensor());

    m.def("_sbvr_init_pool",     &sbvr_init_pool,     py::arg("num_threads"));
    m.def("_sbvr_finalize_pool", &sbvr_finalize_pool);
}