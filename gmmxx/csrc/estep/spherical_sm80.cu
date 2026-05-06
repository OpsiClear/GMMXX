// sm_80+ mma.sync E-step kernels for spherical Gaussian Mixture Models.
//
// Plan 4 Task 5 — SCOPE NOTE
// ===========================
// assign_sm80    : FULLY IMPLEMENTED with mma_m16n8k16 tensor-core cross-product,
//                  cp.async double-buffered centroid tile, and a fp32 GMM-logit
//                  epilogue. Tile: BLOCK_N=128, BLOCK_K=64, BLOCK_D=16.
//                  Mirrors flash-kmeans-cuda assign_sm80.cu adapted for GMM logit.
//
// logsumexp_sm80 : FULLY IMPLEMENTED (Plan 4 Task 5). Same outer template as
//                  assign_sm80; epilogue runs an online (running) logsumexp
//                  across K-chunks, then warp-reduces (max, sumexp) within
//                  each 4-lane row group, writes log_norm = max + log(sumexp).
//                  Output dtype: fp32 (B, N).
//
// resp_sm80    : FULLY IMPLEMENTED (Plan 4 Task 6). Same outer template as
//                assign_sm80; epilogue writes exp(logit - log_norm) per (m, k)
//                directly to out[(b*N + n_global)*K + k]. No cross-warp
//                reduction needed. Output dtype: fp32 (B, N, K).
//
// Fragment register layout (m16n8k16, row.col, fp16/bf16 -> fp32 accumulator):
//
//   A regs (4 u32 / thread, 2 fp16 packed each):
//     a0: A[m = lane/4,     k = 2*(lane%4) .. 2*(lane%4)+1]   (M-half 0, K-half 0)
//     a1: A[m = lane/4 + 8, k = 2*(lane%4) .. 2*(lane%4)+1]   (M-half 1, K-half 0)
//     a2: A[m = lane/4,     k = 2*(lane%4) + 8 .. + 9]        (M-half 0, K-half 1)
//     a3: A[m = lane/4 + 8, k = 2*(lane%4) + 8 .. + 9]        (M-half 1, K-half 1)
//
//   B regs (2 u32 / thread, 2 fp16 packed each — packed along K):
//     b0: B[k = 2*(lane%4) .. + 1, n = lane/4]                 (K-half 0, single N col)
//     b1: B[k = 2*(lane%4) + 8 .. + 9, n = lane/4]             (K-half 1, single N col)
//
//   D regs (4 fp32 / thread):
//     d0: D[m = lane/4,     n = 2*(lane%4)]
//     d1: D[m = lane/4,     n = 2*(lane%4) + 1]
//     d2: D[m = lane/4 + 8, n = 2*(lane%4)]
//     d3: D[m = lane/4 + 8, n = 2*(lane%4) + 1]
//
// Tile layout per CTA:
//   BLOCK_N = 128 points along N (4 warps × WARP_M=32 rows each)
//   BLOCK_K =  64 centroids per K-chunk (N_ATOMS_PER_WARP=8, 8 cols/atom)
//   BLOCK_D =  16 features per mma K-step (2 K-halves × 8 fp16 cols)
//
// SMEM x_tile is loaded once (BLOCK_N × D) per CTA and reused across all K-chunks.
// SMEM c_tile is double-buffered (2 stages × BLOCK_K × D) via cp.async.
// x_sq cache is hoisted to registers after the first sync.
//
// Restriction: D must be a multiple of BLOCK_D=16.

#include "spherical.h"
#include "../common/arch.cuh"
#include "../common/ptx.cuh"
#include "../common/torch_cuda_includes.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

namespace gmmxx { namespace estep { namespace spherical {

namespace {

// ------------------------------------------------------------------
// Tile constants
// ------------------------------------------------------------------
constexpr int BLOCK_D   = 16;   // mma K-dimension (K=16 for m16n8k16)
constexpr int SMEM_PAD  = 8;    // fp16 elts padding per row (avoids bank conflicts on pow2-D)

// ------------------------------------------------------------------
// Per-point running best (maximum logit) stored in registers.
// ------------------------------------------------------------------
struct Best {
    float logit;
    int   k;
};

__device__ __forceinline__ void update_best(Best& b, float logit, int k) {
    if (logit > b.logit) {
        b.logit = logit;
        b.k     = k;
    }
}

// ------------------------------------------------------------------
// cp.async tile helpers — identical pattern to FKC
// ------------------------------------------------------------------

// Issue async loads for one SMEM tile of (rows_max × D) fp16/bf16 elements.
// Source is contiguous; destination has stride D_SMEM (= D + SMEM_PAD).
// Rows >= `rows` are zero-filled (predicate = false).
template <typename T, int THREADS>
__device__ __forceinline__ void async_load_tile(
    T*       smem_tile,
    const T* gmem_tile,
    int      rows,
    int      rows_max,
    int      D,
    int      D_SMEM) {
    const int tid = threadIdx.x;
    const int elts_per_load = 16 / sizeof(T);  // 8 for fp16/bf16
    const int total_elts    = rows_max * D;
    for (int off = tid * elts_per_load; off < total_elts;
         off += THREADS * elts_per_load) {
        int row  = off / D;
        int col  = off % D;
        bool valid = (row < rows);
        T*        dst = smem_tile + (size_t)row * D_SMEM + col;
        const T*  src = gmem_tile + (size_t)row * D       + col;
        ptx::cp_async_16B(ptx::cvta_to_shared(dst), src, valid);
    }
}

// Scalar (non-async) load of c_sq for a K-chunk into a SMEM float array.
// Pads entries beyond k_count with 0.
template <int THREADS>
__device__ __forceinline__ void load_csq_tile(
    float*       smem_csq,
    const float* gmem_csq,
    int          k_count,
    int          k_max) {
    const int tid = threadIdx.x;
    for (int i = tid; i < k_max; i += THREADS) {
        smem_csq[i] = (i < k_count) ? gmem_csq[i] : 0.f;
    }
}

// ------------------------------------------------------------------
// Main kernel
// ------------------------------------------------------------------

template <typename T, int BLOCK_N, int BLOCK_K, int WARPS_PER_CTA>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, 1)
assign_sm80_kernel(
    const T*     __restrict__ x,          // (B, N, D)
    const T*     __restrict__ means,      // (B, K, D)
    const float* __restrict__ var,        // (B, K)
    const float* __restrict__ log_w,      // (B, K)
    const float* __restrict__ x_sq,       // (B, N)
    const float* __restrict__ c_sq,       // (B, K)
    int32_t*     __restrict__ out,        // (B, N)
    int B, int N, int K, int D) {

    constexpr int THREADS_PER_CTA  = WARPS_PER_CTA * 32;
    constexpr int WARP_M           = BLOCK_N / WARPS_PER_CTA;  // rows per warp
    constexpr int M_ATOMS_PER_WARP = WARP_M / 16;              // mma m-atoms per warp
    constexpr int N_ATOMS_PER_WARP = BLOCK_K / 8;              // mma n-atoms per warp (8 n-cols each)
    constexpr int PIPE_STAGES      = 2;

    static_assert(WARP_M >= 16 && (WARP_M % 16) == 0,
                  "WARP_M must be >= 16 and a multiple of 16");
    static_assert((BLOCK_K % 8) == 0, "BLOCK_K must be a multiple of 8");
    static_assert((N_ATOMS_PER_WARP % 2) == 0,
                  "ldmatrix.x4 covers 2 N-atoms; N_ATOMS_PER_WARP must be even");

    const int pid_b   = blockIdx.y;
    const int n_start = blockIdx.x * BLOCK_N;
    const int n_count = min(BLOCK_N, N - n_start);
    if (n_count <= 0) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid / kWarp;
    const int lane    = tid % kWarp;
    const int D_SMEM  = D + SMEM_PAD;

    // ------------------------------------------------------------------
    // SMEM layout:
    //   x_smem  [BLOCK_N * D_SMEM]
    //   c_smem  [PIPE_STAGES * BLOCK_K * D_SMEM]
    //   csq_smem[PIPE_STAGES * BLOCK_K]   (float, after the fp16/bf16 tiles)
    // ------------------------------------------------------------------
    extern __shared__ unsigned char smem_raw[];
    T*     x_smem   = reinterpret_cast<T*>(smem_raw);
    T*     c_smem   = x_smem + (size_t)BLOCK_N * D_SMEM;
    float* csq_smem = reinterpret_cast<float*>(
                          c_smem + (size_t)PIPE_STAGES * BLOCK_K * D_SMEM);

    // ------------------------------------------------------------------
    // Per-thread row/col constants for mma D-register interpretation.
    // D[m = lane/4,     n = 2*(lane%4)]     -> d0
    // D[m = lane/4,     n = 2*(lane%4) + 1] -> d1
    // D[m = lane/4 + 8, n = 2*(lane%4)]     -> d2
    // D[m = lane/4 + 8, n = 2*(lane%4) + 1] -> d3
    // ------------------------------------------------------------------
    const int row_top_in_warp = lane / 4;
    const int row_bot_in_warp = row_top_in_warp + 8;
    const int col_in_atom     = (lane % 4) * 2;

    // ldmatrix lane-mapping constants (from FKC).
    const int ldm_row_off      = (lane & 8)  ? 8 : 0;
    const int ldm_col_off      = (lane & 16) ? 8 : 0;
    const int ldm_row_in_half  = lane & 7;
    const int ldm_n_atom_off   = (lane & 8) ? 8 : 0;

    const int num_k_chunks = (K + BLOCK_K - 1) / BLOCK_K;

    // ------------------------------------------------------------------
    // Load x_smem once (all K-chunks reuse it).
    // ------------------------------------------------------------------
    async_load_tile<T, THREADS_PER_CTA>(
        x_smem,
        x + (size_t)pid_b * N * D + (size_t)n_start * D,
        n_count, BLOCK_N, D, D_SMEM);
    ptx::cp_async_commit();

    // ------------------------------------------------------------------
    // Prime the pipeline: issue chunk 0 (and chunk 1 if PIPE_STAGES=2).
    // ------------------------------------------------------------------
    auto issue_c_chunk = [&](int chunk_idx, int stage) {
        int k_start = chunk_idx * BLOCK_K;
        int k_count = min(BLOCK_K, K - k_start);
        T* c_dst = c_smem + (size_t)stage * BLOCK_K * D_SMEM;
        async_load_tile<T, THREADS_PER_CTA>(
            c_dst,
            means + (size_t)pid_b * K * D + (size_t)k_start * D,
            k_count, BLOCK_K, D, D_SMEM);
        float* csq_dst = csq_smem + stage * BLOCK_K;
        load_csq_tile<THREADS_PER_CTA>(
            csq_dst,
            c_sq + (size_t)pid_b * K + k_start,
            k_count, BLOCK_K);
        ptx::cp_async_commit();
    };

    for (int s = 0; s < PIPE_STAGES && s < num_k_chunks; ++s) {
        issue_c_chunk(s, s);
    }

    // Wait for x_smem and the first c-chunk (PIPE_STAGES-1 still pending).
    ptx::cp_async_wait_group<PIPE_STAGES - 1>();
    __syncthreads();

    // ------------------------------------------------------------------
    // Cache x_sq in registers (one entry per row this warp owns).
    // ------------------------------------------------------------------
    float xs_top_cache[M_ATOMS_PER_WARP];
    float xs_bot_cache[M_ATOMS_PER_WARP];
    bool  top_valid_cache[M_ATOMS_PER_WARP];
    bool  bot_valid_cache[M_ATOMS_PER_WARP];

    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
        int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
        top_valid_cache[m] = (row_top < n_count);
        bot_valid_cache[m] = (row_bot < n_count);
        xs_top_cache[m] = top_valid_cache[m]
            ? x_sq[(size_t)pid_b * N + n_start + row_top] : 0.f;
        xs_bot_cache[m] = bot_valid_cache[m]
            ? x_sq[(size_t)pid_b * N + n_start + row_bot] : 0.f;
    }

    // ------------------------------------------------------------------
    // Per-thread best over all K.
    // ------------------------------------------------------------------
    Best best_top[M_ATOMS_PER_WARP];
    Best best_bot[M_ATOMS_PER_WARP];
    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        best_top[m] = Best{-FLT_MAX, 0};
        best_bot[m] = Best{-FLT_MAX, 0};
    }

    // ------------------------------------------------------------------
    // K-chunk loop.
    // ------------------------------------------------------------------
    for (int chunk_idx = 0; chunk_idx < num_k_chunks; ++chunk_idx) {
        const int k_global_start = chunk_idx * BLOCK_K;
        const int k_count        = min(BLOCK_K, K - k_global_start);
        const int stage          = chunk_idx % PIPE_STAGES;
        T*     c_tile   = c_smem   + (size_t)stage * BLOCK_K * D_SMEM;
        float* csq_tile = csq_smem + stage * BLOCK_K;

        // Per-warp fp32 accumulators [M_ATOMS][N_ATOMS][4].
        // d0 = cross[top, n0], d1 = cross[top, n1],
        // d2 = cross[bot, n0], d3 = cross[bot, n1]
        float acc[M_ATOMS_PER_WARP][N_ATOMS_PER_WARP][4];
        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m)
            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; ++n)
                acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;

        // ------ D-dimension mma loop ------
        #pragma unroll 4
        for (int d_off = 0; d_off < D; d_off += BLOCK_D) {

            // -- Load A regs (x_smem -> mma A) via ldmatrix.x4 --
            // One x4 per m-atom covers a 16×16 sub-tile at (m_base, d_off).
            uint32_t a_regs[M_ATOMS_PER_WARP][4];
            #pragma unroll
            for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
                int m_base = warp_id * WARP_M + m * 16;
                int row    = m_base + ldm_row_off + ldm_row_in_half;
                unsigned int smem_addr = ptx::cvta_to_shared(
                    x_smem + (size_t)row * D_SMEM + d_off + ldm_col_off);
                ptx::ldmatrix_x4(a_regs[m][0], a_regs[m][1],
                                 a_regs[m][2], a_regs[m][3], smem_addr);
            }

            // -- Load B regs (c_tile -> mma B) via ldmatrix.x4, 2 atoms at a time --
            // c_tile is (BLOCK_K, D_SMEM), row-major: row = centroid index, col = D.
            // ldmatrix.x4 (no-trans) maps 4 matrices:
            //   matrix 0: atom n,   K-half 0  -> b_regs[n][0]
            //   matrix 1: atom n+1, K-half 0  -> b_regs[n+1][0]
            //   matrix 2: atom n,   K-half 1  -> b_regs[n][1]
            //   matrix 3: atom n+1, K-half 1  -> b_regs[n+1][1]
            uint32_t b_regs[N_ATOMS_PER_WARP][2];
            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; n += 2) {
                int n_col = n * 8 + ldm_n_atom_off + ldm_row_in_half;
                unsigned int smem_addr = ptx::cvta_to_shared(
                    c_tile + (size_t)n_col * D_SMEM + d_off + ldm_col_off);
                uint32_t r0, r1, r2, r3;
                ptx::ldmatrix_x4(r0, r1, r2, r3, smem_addr);
                b_regs[n    ][0] = r0;
                b_regs[n + 1][0] = r1;
                b_regs[n    ][1] = r2;
                b_regs[n + 1][1] = r3;
            }

            // -- Issue mma atoms (fp32 acc) --
            #pragma unroll
            for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
                #pragma unroll
                for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
                    if constexpr (sizeof(T) == 2) {
                        // Both fp16 and bf16 use 2-byte elements; dispatch by
                        // checking actual T at compile time via a tag trait.
                    }
                    // We always use fp32 accumulator for GMM logit correctness.
                    // The mma_m16n8k16_fp16 / mma_m16n8k16_bf16 both produce fp32 acc.
                    // We pick based on T == __half vs __nv_bfloat16.
                    if constexpr (std::is_same<T, __half>::value ||
                                  std::is_same<T, at::Half>::value) {
                        ptx::mma_m16n8k16_fp16(
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3],
                            a_regs[m][0], a_regs[m][1],
                            a_regs[m][2], a_regs[m][3],
                            b_regs[n][0], b_regs[n][1],
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3]);
                    } else {
                        ptx::mma_m16n8k16_bf16(
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3],
                            a_regs[m][0], a_regs[m][1],
                            a_regs[m][2], a_regs[m][3],
                            b_regs[n][0], b_regs[n][1],
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3]);
                    }
                }
            }
        }  // d-loop

        // ------ Epilogue: GMM logit and best-k update ------
        // acc[m][n] holds cross-products x·c for the (m, n) tile fragment.
        // GMM logit:  logit_k = log_w_k - 0.5*D*log(2π*v_k) - 0.5/v_k * (xs + cs - 2*cross)
        // We read var, log_w, c_sq from global memory (scalar, K-indexed).

        const float* var_b   = var   + (size_t)pid_b * K;
        const float* log_w_b = log_w + (size_t)pid_b * K;
        const float inv_two_pi_log = 0.5f * 1.8378770664f;  // 0.5 * log(2π)
        const float half_D_log2pi  = 0.5f * (float)D * 1.8378770664f;  // 0.5*D*log(2π)

        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            const float xs_top = xs_top_cache[m];
            const float xs_bot = xs_bot_cache[m];
            const bool  tv     = top_valid_cache[m];
            const bool  bv     = bot_valid_cache[m];

            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
                // Two global k-indices this (m, n, lane) covers:
                int k_in_chunk_0 = n * 8 + col_in_atom;
                int k_in_chunk_1 = k_in_chunk_0 + 1;
                int k0 = k_global_start + k_in_chunk_0;
                int k1 = k_global_start + k_in_chunk_1;
                bool k0v = (k0 < K);
                bool k1v = (k1 < K);

                // acc d-register layout:
                //   acc[m][n][0] = cross[top, k0]  (d0 = D[m=top, n=2*(lane%4)])
                //   acc[m][n][1] = cross[top, k1]  (d1 = D[m=top, n=2*(lane%4)+1])
                //   acc[m][n][2] = cross[bot, k0]  (d2 = D[m=bot, n=2*(lane%4)])
                //   acc[m][n][3] = cross[bot, k1]  (d3 = D[m=bot, n=2*(lane%4)+1])
                float cross_top0 = acc[m][n][0];
                float cross_top1 = acc[m][n][1];
                float cross_bot0 = acc[m][n][2];
                float cross_bot1 = acc[m][n][3];

                float cs0 = csq_tile[k_in_chunk_0];
                float cs1 = csq_tile[k_in_chunk_1];

                if (tv && k0v) {
                    float v0      = var_b[k0];
                    float dist0   = xs_top + cs0 - 2.f * cross_top0;
                    if (dist0 < 0.f) dist0 = 0.f;
                    float logit0  = log_w_b[k0]
                                  - half_D_log2pi
                                  - 0.5f * (float)D * logf(v0)
                                  - 0.5f * dist0 / v0;
                    update_best(best_top[m], logit0, k0);
                }
                if (tv && k1v) {
                    float v1      = var_b[k1];
                    float dist1   = xs_top + cs1 - 2.f * cross_top1;
                    if (dist1 < 0.f) dist1 = 0.f;
                    float logit1  = log_w_b[k1]
                                  - half_D_log2pi
                                  - 0.5f * (float)D * logf(v1)
                                  - 0.5f * dist1 / v1;
                    update_best(best_top[m], logit1, k1);
                }
                if (bv && k0v) {
                    float v0      = var_b[k0];
                    float dist0   = xs_bot + cs0 - 2.f * cross_bot0;
                    if (dist0 < 0.f) dist0 = 0.f;
                    float logit0  = log_w_b[k0]
                                  - half_D_log2pi
                                  - 0.5f * (float)D * logf(v0)
                                  - 0.5f * dist0 / v0;
                    update_best(best_bot[m], logit0, k0);
                }
                if (bv && k1v) {
                    float v1      = var_b[k1];
                    float dist1   = xs_bot + cs1 - 2.f * cross_bot1;
                    if (dist1 < 0.f) dist1 = 0.f;
                    float logit1  = log_w_b[k1]
                                  - half_D_log2pi
                                  - 0.5f * (float)D * logf(v1)
                                  - 0.5f * dist1 / v1;
                    update_best(best_bot[m], logit1, k1);
                }
            }
        }  // m-loop

        // Prefetch the chunk that will be consumed PIPE_STAGES iterations later.
        int prefetch_idx = chunk_idx + PIPE_STAGES;
        if (prefetch_idx < num_k_chunks) {
            issue_c_chunk(prefetch_idx, prefetch_idx % PIPE_STAGES);
        }
        if (chunk_idx + 1 < num_k_chunks) {
            ptx::cp_async_wait_group<PIPE_STAGES - 1>();
            __syncthreads();
        }
    }  // k-chunk loop

    // ------------------------------------------------------------------
    // Warp-level reduction: 4 lanes in a sub-group share the same row.
    // lane/4 selects row_top; each lane holds candidates for n-cols 0,2,4,6
    // + their +1. Reduce within the 4-lane group.
    // ------------------------------------------------------------------
    auto warp_reduce_best = [&](Best& b) {
        #pragma unroll
        for (int offset : {1, 2}) {
            float other_logit = __shfl_xor_sync(0xffffffff, b.logit, offset, 4);
            int   other_k     = __shfl_xor_sync(0xffffffff, b.k,     offset, 4);
            if (other_logit > b.logit ||
                (other_logit == b.logit && other_k < b.k)) {
                b.logit = other_logit;
                b.k     = other_k;
            }
        }
    };

    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        warp_reduce_best(best_top[m]);
        warp_reduce_best(best_bot[m]);
    }

    // Lane (lane % 4 == 0) holds the row's answer.
    if ((lane % 4) == 0) {
        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
            int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
            if (row_top < n_count) {
                out[(size_t)pid_b * N + n_start + row_top] = best_top[m].k;
            }
            if (row_bot < n_count) {
                out[(size_t)pid_b * N + n_start + row_bot] = best_bot[m].k;
            }
        }
    }
}  // assign_sm80_kernel

// ------------------------------------------------------------------
// logsumexp_sm80 kernel — same outer template as assign_sm80, but the
// epilogue runs an online (running) logsumexp across K-chunks instead
// of a running argmax.
//
// Per-thread per-row state: (max_so_far, sumexp) as fp32 registers.
// Update on each new logit `v`:
//   if v > max_so_far:
//       sumexp = sumexp * exp(max_so_far - v) + 1
//       max_so_far = v
//   else:
//       sumexp += exp(v - max_so_far)
//
// Final reduction across the 4 lanes that share a row uses warp shuffles
// to merge (max, sumexp) pairs stably:
//   M = max(m1, m2)
//   S = s1*exp(m1-M) + s2*exp(m2-M)
// Lane (lane%4 == 0) writes log_norm[m] = max_so_far + log(sumexp).
// ------------------------------------------------------------------
struct LSE {
    float m;   // running max
    float s;   // running sumexp (relative to m)
};

// Fold a single new logit `v` into running (m, s).
__device__ __forceinline__ void update_lse(LSE& l, float v) {
    if (v > l.m) {
        // Rescale old sumexp; previous m is now smaller than v.
        // exp(old_m - v) is in (0, 1], safe.
        l.s = l.s * __expf(l.m - v) + 1.f;
        l.m = v;
    } else {
        l.s += __expf(v - l.m);
    }
}

template <typename T, int BLOCK_N, int BLOCK_K, int WARPS_PER_CTA>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, 1)
logsumexp_sm80_kernel(
    const T*     __restrict__ x,          // (B, N, D)
    const T*     __restrict__ means,      // (B, K, D)
    const float* __restrict__ var,        // (B, K)
    const float* __restrict__ log_w,      // (B, K)
    const float* __restrict__ x_sq,       // (B, N)
    const float* __restrict__ c_sq,       // (B, K)
    float*       __restrict__ out,        // (B, N)  fp32 log-norm
    int B, int N, int K, int D) {

    constexpr int THREADS_PER_CTA  = WARPS_PER_CTA * 32;
    constexpr int WARP_M           = BLOCK_N / WARPS_PER_CTA;
    constexpr int M_ATOMS_PER_WARP = WARP_M / 16;
    constexpr int N_ATOMS_PER_WARP = BLOCK_K / 8;
    constexpr int PIPE_STAGES      = 2;

    static_assert(WARP_M >= 16 && (WARP_M % 16) == 0,
                  "WARP_M must be >= 16 and a multiple of 16");
    static_assert((BLOCK_K % 8) == 0, "BLOCK_K must be a multiple of 8");
    static_assert((N_ATOMS_PER_WARP % 2) == 0,
                  "ldmatrix.x4 covers 2 N-atoms; N_ATOMS_PER_WARP must be even");

    const int pid_b   = blockIdx.y;
    const int n_start = blockIdx.x * BLOCK_N;
    const int n_count = min(BLOCK_N, N - n_start);
    if (n_count <= 0) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid / kWarp;
    const int lane    = tid % kWarp;
    const int D_SMEM  = D + SMEM_PAD;

    extern __shared__ unsigned char smem_raw[];
    T*     x_smem   = reinterpret_cast<T*>(smem_raw);
    T*     c_smem   = x_smem + (size_t)BLOCK_N * D_SMEM;
    float* csq_smem = reinterpret_cast<float*>(
                          c_smem + (size_t)PIPE_STAGES * BLOCK_K * D_SMEM);

    const int row_top_in_warp = lane / 4;
    const int row_bot_in_warp = row_top_in_warp + 8;
    const int col_in_atom     = (lane % 4) * 2;

    const int ldm_row_off      = (lane & 8)  ? 8 : 0;
    const int ldm_col_off      = (lane & 16) ? 8 : 0;
    const int ldm_row_in_half  = lane & 7;
    const int ldm_n_atom_off   = (lane & 8) ? 8 : 0;

    const int num_k_chunks = (K + BLOCK_K - 1) / BLOCK_K;

    // Load x_smem once.
    async_load_tile<T, THREADS_PER_CTA>(
        x_smem,
        x + (size_t)pid_b * N * D + (size_t)n_start * D,
        n_count, BLOCK_N, D, D_SMEM);
    ptx::cp_async_commit();

    auto issue_c_chunk = [&](int chunk_idx, int stage) {
        int k_start = chunk_idx * BLOCK_K;
        int k_count = min(BLOCK_K, K - k_start);
        T* c_dst = c_smem + (size_t)stage * BLOCK_K * D_SMEM;
        async_load_tile<T, THREADS_PER_CTA>(
            c_dst,
            means + (size_t)pid_b * K * D + (size_t)k_start * D,
            k_count, BLOCK_K, D, D_SMEM);
        float* csq_dst = csq_smem + stage * BLOCK_K;
        load_csq_tile<THREADS_PER_CTA>(
            csq_dst,
            c_sq + (size_t)pid_b * K + k_start,
            k_count, BLOCK_K);
        ptx::cp_async_commit();
    };

    for (int s = 0; s < PIPE_STAGES && s < num_k_chunks; ++s) {
        issue_c_chunk(s, s);
    }

    ptx::cp_async_wait_group<PIPE_STAGES - 1>();
    __syncthreads();

    // Cache x_sq in registers (one entry per row this warp owns).
    float xs_top_cache[M_ATOMS_PER_WARP];
    float xs_bot_cache[M_ATOMS_PER_WARP];
    bool  top_valid_cache[M_ATOMS_PER_WARP];
    bool  bot_valid_cache[M_ATOMS_PER_WARP];

    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
        int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
        top_valid_cache[m] = (row_top < n_count);
        bot_valid_cache[m] = (row_bot < n_count);
        xs_top_cache[m] = top_valid_cache[m]
            ? x_sq[(size_t)pid_b * N + n_start + row_top] : 0.f;
        xs_bot_cache[m] = bot_valid_cache[m]
            ? x_sq[(size_t)pid_b * N + n_start + row_bot] : 0.f;
    }

    // Per-thread running (max, sumexp) for each row this thread covers.
    LSE lse_top[M_ATOMS_PER_WARP];
    LSE lse_bot[M_ATOMS_PER_WARP];
    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        lse_top[m] = LSE{-FLT_MAX, 0.f};
        lse_bot[m] = LSE{-FLT_MAX, 0.f};
    }

    for (int chunk_idx = 0; chunk_idx < num_k_chunks; ++chunk_idx) {
        const int k_global_start = chunk_idx * BLOCK_K;
        const int stage          = chunk_idx % PIPE_STAGES;
        T*     c_tile   = c_smem   + (size_t)stage * BLOCK_K * D_SMEM;
        float* csq_tile = csq_smem + stage * BLOCK_K;

        float acc[M_ATOMS_PER_WARP][N_ATOMS_PER_WARP][4];
        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m)
            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; ++n)
                acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;

        #pragma unroll 4
        for (int d_off = 0; d_off < D; d_off += BLOCK_D) {

            uint32_t a_regs[M_ATOMS_PER_WARP][4];
            #pragma unroll
            for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
                int m_base = warp_id * WARP_M + m * 16;
                int row    = m_base + ldm_row_off + ldm_row_in_half;
                unsigned int smem_addr = ptx::cvta_to_shared(
                    x_smem + (size_t)row * D_SMEM + d_off + ldm_col_off);
                ptx::ldmatrix_x4(a_regs[m][0], a_regs[m][1],
                                 a_regs[m][2], a_regs[m][3], smem_addr);
            }

            uint32_t b_regs[N_ATOMS_PER_WARP][2];
            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; n += 2) {
                int n_col = n * 8 + ldm_n_atom_off + ldm_row_in_half;
                unsigned int smem_addr = ptx::cvta_to_shared(
                    c_tile + (size_t)n_col * D_SMEM + d_off + ldm_col_off);
                uint32_t r0, r1, r2, r3;
                ptx::ldmatrix_x4(r0, r1, r2, r3, smem_addr);
                b_regs[n    ][0] = r0;
                b_regs[n + 1][0] = r1;
                b_regs[n    ][1] = r2;
                b_regs[n + 1][1] = r3;
            }

            #pragma unroll
            for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
                #pragma unroll
                for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
                    if constexpr (std::is_same<T, __half>::value ||
                                  std::is_same<T, at::Half>::value) {
                        ptx::mma_m16n8k16_fp16(
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3],
                            a_regs[m][0], a_regs[m][1],
                            a_regs[m][2], a_regs[m][3],
                            b_regs[n][0], b_regs[n][1],
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3]);
                    } else {
                        ptx::mma_m16n8k16_bf16(
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3],
                            a_regs[m][0], a_regs[m][1],
                            a_regs[m][2], a_regs[m][3],
                            b_regs[n][0], b_regs[n][1],
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3]);
                    }
                }
            }
        }  // d-loop

        // ------ Epilogue: GMM logit + running logsumexp ------
        const float* var_b   = var   + (size_t)pid_b * K;
        const float* log_w_b = log_w + (size_t)pid_b * K;
        const float half_D_log2pi = 0.5f * (float)D * 1.8378770664f;

        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            const float xs_top = xs_top_cache[m];
            const float xs_bot = xs_bot_cache[m];
            const bool  tv     = top_valid_cache[m];
            const bool  bv     = bot_valid_cache[m];

            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
                int k_in_chunk_0 = n * 8 + col_in_atom;
                int k_in_chunk_1 = k_in_chunk_0 + 1;
                int k0 = k_global_start + k_in_chunk_0;
                int k1 = k_global_start + k_in_chunk_1;
                bool k0v = (k0 < K);
                bool k1v = (k1 < K);

                float cross_top0 = acc[m][n][0];
                float cross_top1 = acc[m][n][1];
                float cross_bot0 = acc[m][n][2];
                float cross_bot1 = acc[m][n][3];

                float cs0 = csq_tile[k_in_chunk_0];
                float cs1 = csq_tile[k_in_chunk_1];

                if (tv && k0v) {
                    float v0      = var_b[k0];
                    float dist0   = xs_top + cs0 - 2.f * cross_top0;
                    if (dist0 < 0.f) dist0 = 0.f;
                    float logit0  = log_w_b[k0]
                                  - half_D_log2pi
                                  - 0.5f * (float)D * logf(v0)
                                  - 0.5f * dist0 / v0;
                    update_lse(lse_top[m], logit0);
                }
                if (tv && k1v) {
                    float v1      = var_b[k1];
                    float dist1   = xs_top + cs1 - 2.f * cross_top1;
                    if (dist1 < 0.f) dist1 = 0.f;
                    float logit1  = log_w_b[k1]
                                  - half_D_log2pi
                                  - 0.5f * (float)D * logf(v1)
                                  - 0.5f * dist1 / v1;
                    update_lse(lse_top[m], logit1);
                }
                if (bv && k0v) {
                    float v0      = var_b[k0];
                    float dist0   = xs_bot + cs0 - 2.f * cross_bot0;
                    if (dist0 < 0.f) dist0 = 0.f;
                    float logit0  = log_w_b[k0]
                                  - half_D_log2pi
                                  - 0.5f * (float)D * logf(v0)
                                  - 0.5f * dist0 / v0;
                    update_lse(lse_bot[m], logit0);
                }
                if (bv && k1v) {
                    float v1      = var_b[k1];
                    float dist1   = xs_bot + cs1 - 2.f * cross_bot1;
                    if (dist1 < 0.f) dist1 = 0.f;
                    float logit1  = log_w_b[k1]
                                  - half_D_log2pi
                                  - 0.5f * (float)D * logf(v1)
                                  - 0.5f * dist1 / v1;
                    update_lse(lse_bot[m], logit1);
                }
            }
        }

        int prefetch_idx = chunk_idx + PIPE_STAGES;
        if (prefetch_idx < num_k_chunks) {
            issue_c_chunk(prefetch_idx, prefetch_idx % PIPE_STAGES);
        }
        if (chunk_idx + 1 < num_k_chunks) {
            ptx::cp_async_wait_group<PIPE_STAGES - 1>();
            __syncthreads();
        }
    }  // k-chunk loop

    // ------------------------------------------------------------------
    // Reduce (m, s) across the 4 lanes that share a row.
    // Stable merge:
    //   M = max(ma, mb)
    //   S = sa*exp(ma - M) + sb*exp(mb - M)
    // ------------------------------------------------------------------
    auto warp_reduce_lse = [&](LSE& l) {
        #pragma unroll
        for (int offset : {1, 2}) {
            float other_m = __shfl_xor_sync(0xffffffff, l.m, offset, 4);
            float other_s = __shfl_xor_sync(0xffffffff, l.s, offset, 4);
            float new_m   = fmaxf(l.m, other_m);
            // Guard against -inf - -inf = nan: if both are -FLT_MAX/-inf, sum is 0.
            float a = (l.m   == -FLT_MAX) ? 0.f : l.s * __expf(l.m   - new_m);
            float b = (other_m == -FLT_MAX) ? 0.f : other_s * __expf(other_m - new_m);
            l.m = new_m;
            l.s = a + b;
        }
    };

    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        warp_reduce_lse(lse_top[m]);
        warp_reduce_lse(lse_bot[m]);
    }

    if ((lane % 4) == 0) {
        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
            int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
            if (row_top < n_count) {
                float val = (lse_top[m].s > 0.f)
                          ? (lse_top[m].m + logf(lse_top[m].s))
                          : -INFINITY;
                out[(size_t)pid_b * N + n_start + row_top] = val;
            }
            if (row_bot < n_count) {
                float val = (lse_bot[m].s > 0.f)
                          ? (lse_bot[m].m + logf(lse_bot[m].s))
                          : -INFINITY;
                out[(size_t)pid_b * N + n_start + row_bot] = val;
            }
        }
    }
}  // logsumexp_sm80_kernel

// ------------------------------------------------------------------
// resp_sm80 kernel — same outer template as assign/logsumexp, but the
// epilogue writes exp(logit - log_norm[b, n_global]) per (m, k) directly
// to the output buffer out[(b*N + n_global)*K + k].
//
// No per-thread running state across K-chunks and no warp reduction —
// each (m, k) maps to a unique output slot.
// log_norm is (B, N) fp32, read once per row at kernel entry into regs.
// out is (B, N, K) fp32.
// ------------------------------------------------------------------
template <typename T, int BLOCK_N, int BLOCK_K, int WARPS_PER_CTA>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, 1)
resp_sm80_kernel(
    const T*     __restrict__ x,          // (B, N, D)
    const T*     __restrict__ means,      // (B, K, D)
    const float* __restrict__ var,        // (B, K)
    const float* __restrict__ log_w,      // (B, K)
    const float* __restrict__ x_sq,       // (B, N)
    const float* __restrict__ c_sq,       // (B, K)
    const float* __restrict__ log_norm,   // (B, N)
    float*       __restrict__ out,        // (B, N, K) fp32
    int B, int N, int K, int D) {

    constexpr int THREADS_PER_CTA  = WARPS_PER_CTA * 32;
    constexpr int WARP_M           = BLOCK_N / WARPS_PER_CTA;
    constexpr int M_ATOMS_PER_WARP = WARP_M / 16;
    constexpr int N_ATOMS_PER_WARP = BLOCK_K / 8;
    constexpr int PIPE_STAGES      = 2;

    static_assert(WARP_M >= 16 && (WARP_M % 16) == 0,
                  "WARP_M must be >= 16 and a multiple of 16");
    static_assert((BLOCK_K % 8) == 0, "BLOCK_K must be a multiple of 8");
    static_assert((N_ATOMS_PER_WARP % 2) == 0,
                  "ldmatrix.x4 covers 2 N-atoms; N_ATOMS_PER_WARP must be even");

    const int pid_b   = blockIdx.y;
    const int n_start = blockIdx.x * BLOCK_N;
    const int n_count = min(BLOCK_N, N - n_start);
    if (n_count <= 0) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid / kWarp;
    const int lane    = tid % kWarp;
    const int D_SMEM  = D + SMEM_PAD;

    extern __shared__ unsigned char smem_raw[];
    T*     x_smem   = reinterpret_cast<T*>(smem_raw);
    T*     c_smem   = x_smem + (size_t)BLOCK_N * D_SMEM;
    float* csq_smem = reinterpret_cast<float*>(
                          c_smem + (size_t)PIPE_STAGES * BLOCK_K * D_SMEM);

    const int row_top_in_warp = lane / 4;
    const int row_bot_in_warp = row_top_in_warp + 8;
    const int col_in_atom     = (lane % 4) * 2;

    const int ldm_row_off      = (lane & 8)  ? 8 : 0;
    const int ldm_col_off      = (lane & 16) ? 8 : 0;
    const int ldm_row_in_half  = lane & 7;
    const int ldm_n_atom_off   = (lane & 8) ? 8 : 0;

    const int num_k_chunks = (K + BLOCK_K - 1) / BLOCK_K;

    // Load x_smem once.
    async_load_tile<T, THREADS_PER_CTA>(
        x_smem,
        x + (size_t)pid_b * N * D + (size_t)n_start * D,
        n_count, BLOCK_N, D, D_SMEM);
    ptx::cp_async_commit();

    auto issue_c_chunk = [&](int chunk_idx, int stage) {
        int k_start = chunk_idx * BLOCK_K;
        int k_count = min(BLOCK_K, K - k_start);
        T* c_dst = c_smem + (size_t)stage * BLOCK_K * D_SMEM;
        async_load_tile<T, THREADS_PER_CTA>(
            c_dst,
            means + (size_t)pid_b * K * D + (size_t)k_start * D,
            k_count, BLOCK_K, D, D_SMEM);
        float* csq_dst = csq_smem + stage * BLOCK_K;
        load_csq_tile<THREADS_PER_CTA>(
            csq_dst,
            c_sq + (size_t)pid_b * K + k_start,
            k_count, BLOCK_K);
        ptx::cp_async_commit();
    };

    for (int s = 0; s < PIPE_STAGES && s < num_k_chunks; ++s) {
        issue_c_chunk(s, s);
    }

    ptx::cp_async_wait_group<PIPE_STAGES - 1>();
    __syncthreads();

    // Cache x_sq in registers (one entry per row this warp owns).
    float xs_top_cache[M_ATOMS_PER_WARP];
    float xs_bot_cache[M_ATOMS_PER_WARP];
    bool  top_valid_cache[M_ATOMS_PER_WARP];
    bool  bot_valid_cache[M_ATOMS_PER_WARP];

    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
        int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
        top_valid_cache[m] = (row_top < n_count);
        bot_valid_cache[m] = (row_bot < n_count);
        xs_top_cache[m] = top_valid_cache[m]
            ? x_sq[(size_t)pid_b * N + n_start + row_top] : 0.f;
        xs_bot_cache[m] = bot_valid_cache[m]
            ? x_sq[(size_t)pid_b * N + n_start + row_bot] : 0.f;
    }

    // Cache log_norm for each row this thread owns (read once from global).
    float ln_top_cache[M_ATOMS_PER_WARP];
    float ln_bot_cache[M_ATOMS_PER_WARP];

    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
        int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
        ln_top_cache[m] = top_valid_cache[m]
            ? log_norm[(size_t)pid_b * N + n_start + row_top] : 0.f;
        ln_bot_cache[m] = bot_valid_cache[m]
            ? log_norm[(size_t)pid_b * N + n_start + row_bot] : 0.f;
    }

    // K-chunk loop.
    for (int chunk_idx = 0; chunk_idx < num_k_chunks; ++chunk_idx) {
        const int k_global_start = chunk_idx * BLOCK_K;
        const int stage          = chunk_idx % PIPE_STAGES;
        T*     c_tile   = c_smem   + (size_t)stage * BLOCK_K * D_SMEM;
        float* csq_tile = csq_smem + stage * BLOCK_K;

        float acc[M_ATOMS_PER_WARP][N_ATOMS_PER_WARP][4];
        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m)
            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; ++n)
                acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;

        // D-dimension mma loop — identical to assign/logsumexp.
        #pragma unroll 4
        for (int d_off = 0; d_off < D; d_off += BLOCK_D) {

            uint32_t a_regs[M_ATOMS_PER_WARP][4];
            #pragma unroll
            for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
                int m_base = warp_id * WARP_M + m * 16;
                int row    = m_base + ldm_row_off + ldm_row_in_half;
                unsigned int smem_addr = ptx::cvta_to_shared(
                    x_smem + (size_t)row * D_SMEM + d_off + ldm_col_off);
                ptx::ldmatrix_x4(a_regs[m][0], a_regs[m][1],
                                 a_regs[m][2], a_regs[m][3], smem_addr);
            }

            uint32_t b_regs[N_ATOMS_PER_WARP][2];
            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; n += 2) {
                int n_col = n * 8 + ldm_n_atom_off + ldm_row_in_half;
                unsigned int smem_addr = ptx::cvta_to_shared(
                    c_tile + (size_t)n_col * D_SMEM + d_off + ldm_col_off);
                uint32_t r0, r1, r2, r3;
                ptx::ldmatrix_x4(r0, r1, r2, r3, smem_addr);
                b_regs[n    ][0] = r0;
                b_regs[n + 1][0] = r1;
                b_regs[n    ][1] = r2;
                b_regs[n + 1][1] = r3;
            }

            #pragma unroll
            for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
                #pragma unroll
                for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
                    if constexpr (std::is_same<T, __half>::value ||
                                  std::is_same<T, at::Half>::value) {
                        ptx::mma_m16n8k16_fp16(
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3],
                            a_regs[m][0], a_regs[m][1],
                            a_regs[m][2], a_regs[m][3],
                            b_regs[n][0], b_regs[n][1],
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3]);
                    } else {
                        ptx::mma_m16n8k16_bf16(
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3],
                            a_regs[m][0], a_regs[m][1],
                            a_regs[m][2], a_regs[m][3],
                            b_regs[n][0], b_regs[n][1],
                            acc[m][n][0], acc[m][n][1],
                            acc[m][n][2], acc[m][n][3]);
                    }
                }
            }
        }  // d-loop

        // ------ Epilogue: compute logit, subtract log_norm, exponentiate,
        //                  write directly to out[(b*N + n_global)*K + k].
        //
        // Each (m, n, lane) owns two distinct (n_global, k) pairs:
        //   top: (n_start + warp_id*WARP_M + m*16 + row_top_in_warp, k_global_start + n*8 + col_in_atom + 0)
        //   bot: same but row = +8
        // No cross-thread communication needed — each k maps to a unique slot.
        // ------------------------------------------------------------------
        const float* var_b   = var   + (size_t)pid_b * K;
        const float* log_w_b = log_w + (size_t)pid_b * K;
        const float half_D_log2pi = 0.5f * (float)D * 1.8378770664f;

        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            const float xs_top = xs_top_cache[m];
            const float xs_bot = xs_bot_cache[m];
            const bool  tv     = top_valid_cache[m];
            const bool  bv     = bot_valid_cache[m];
            const float ln_top = ln_top_cache[m];
            const float ln_bot = ln_bot_cache[m];

            // Global row indices for this (m, warp) pair.
            const int row_top_global = n_start + warp_id * WARP_M + m * 16 + row_top_in_warp;
            const int row_bot_global = n_start + warp_id * WARP_M + m * 16 + row_bot_in_warp;

            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
                int k_in_chunk_0 = n * 8 + col_in_atom;
                int k_in_chunk_1 = k_in_chunk_0 + 1;
                int k0 = k_global_start + k_in_chunk_0;
                int k1 = k_global_start + k_in_chunk_1;
                bool k0v = (k0 < K);
                bool k1v = (k1 < K);

                float cross_top0 = acc[m][n][0];
                float cross_top1 = acc[m][n][1];
                float cross_bot0 = acc[m][n][2];
                float cross_bot1 = acc[m][n][3];

                float cs0 = csq_tile[k_in_chunk_0];
                float cs1 = csq_tile[k_in_chunk_1];

                if (tv && k0v) {
                    float v0    = var_b[k0];
                    float dist0 = xs_top + cs0 - 2.f * cross_top0;
                    if (dist0 < 0.f) dist0 = 0.f;
                    float logit0 = log_w_b[k0]
                                 - half_D_log2pi
                                 - 0.5f * (float)D * logf(v0)
                                 - 0.5f * dist0 / v0;
                    out[((size_t)pid_b * N + row_top_global) * K + k0] =
                        __expf(logit0 - ln_top);
                }
                if (tv && k1v) {
                    float v1    = var_b[k1];
                    float dist1 = xs_top + cs1 - 2.f * cross_top1;
                    if (dist1 < 0.f) dist1 = 0.f;
                    float logit1 = log_w_b[k1]
                                 - half_D_log2pi
                                 - 0.5f * (float)D * logf(v1)
                                 - 0.5f * dist1 / v1;
                    out[((size_t)pid_b * N + row_top_global) * K + k1] =
                        __expf(logit1 - ln_top);
                }
                if (bv && k0v) {
                    float v0    = var_b[k0];
                    float dist0 = xs_bot + cs0 - 2.f * cross_bot0;
                    if (dist0 < 0.f) dist0 = 0.f;
                    float logit0 = log_w_b[k0]
                                 - half_D_log2pi
                                 - 0.5f * (float)D * logf(v0)
                                 - 0.5f * dist0 / v0;
                    out[((size_t)pid_b * N + row_bot_global) * K + k0] =
                        __expf(logit0 - ln_bot);
                }
                if (bv && k1v) {
                    float v1    = var_b[k1];
                    float dist1 = xs_bot + cs1 - 2.f * cross_bot1;
                    if (dist1 < 0.f) dist1 = 0.f;
                    float logit1 = log_w_b[k1]
                                 - half_D_log2pi
                                 - 0.5f * (float)D * logf(v1)
                                 - 0.5f * dist1 / v1;
                    out[((size_t)pid_b * N + row_bot_global) * K + k1] =
                        __expf(logit1 - ln_bot);
                }
            }
        }  // m-loop

        int prefetch_idx = chunk_idx + PIPE_STAGES;
        if (prefetch_idx < num_k_chunks) {
            issue_c_chunk(prefetch_idx, prefetch_idx % PIPE_STAGES);
        }
        if (chunk_idx + 1 < num_k_chunks) {
            ptx::cp_async_wait_group<PIPE_STAGES - 1>();
            __syncthreads();
        }
    }  // k-chunk loop
    // No final reduction — all writes done in-place in the epilogue above.
}  // resp_sm80_kernel

// ------------------------------------------------------------------
// logsumexp_resp_sm80_kernel — fused logsumexp + resp.
//
// Same outer template as logsumexp_sm80; runs ONE GEMM pass, caches per-row
// per-K logits in registers, then in a second epilogue pass computes
// resp[n,k] = exp(logit[n,k] - lse[n]) and writes both lse[n] and resp[n,k].
//
// Constraint: K must be <= BLOCK_K (single K-chunk). The dispatcher gates
// this so callers never trip the assertion path.
//
// Compared to calling logsumexp_sm80 + resp_sm80 separately, this saves
// one full GEMM-equivalent of work (each kernel currently runs the full
// GEMM independently) and one kernel launch.
// ------------------------------------------------------------------
template <typename T, int BLOCK_N, int BLOCK_K, int WARPS_PER_CTA>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, 1)
logsumexp_resp_sm80_kernel(
    const T*     __restrict__ x,          // (B, N, D)
    const T*     __restrict__ means,      // (B, K, D)
    const float* __restrict__ var,        // (B, K)
    const float* __restrict__ log_w,      // (B, K)
    const float* __restrict__ x_sq,       // (B, N)
    const float* __restrict__ c_sq,       // (B, K)
    float*       __restrict__ lse_out,    // (B, N)  fp32 — may be nullptr
    float*       __restrict__ resp_out,   // (B, N, K) fp32
    int B, int N, int K, int D) {

    constexpr int THREADS_PER_CTA  = WARPS_PER_CTA * 32;
    constexpr int WARP_M           = BLOCK_N / WARPS_PER_CTA;
    constexpr int M_ATOMS_PER_WARP = WARP_M / 16;
    constexpr int N_ATOMS_PER_WARP = BLOCK_K / 8;
    constexpr int PIPE_STAGES      = 2;

    static_assert(WARP_M >= 16 && (WARP_M % 16) == 0,
                  "WARP_M must be >= 16 and a multiple of 16");
    static_assert((BLOCK_K % 8) == 0, "BLOCK_K must be a multiple of 8");
    static_assert((N_ATOMS_PER_WARP % 2) == 0,
                  "ldmatrix.x4 covers 2 N-atoms; N_ATOMS_PER_WARP must be even");

    const int pid_b   = blockIdx.y;
    const int n_start = blockIdx.x * BLOCK_N;
    const int n_count = min(BLOCK_N, N - n_start);
    if (n_count <= 0) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid / kWarp;
    const int lane    = tid % kWarp;
    const int D_SMEM  = D + SMEM_PAD;

    extern __shared__ unsigned char smem_raw[];
    T*     x_smem   = reinterpret_cast<T*>(smem_raw);
    T*     c_smem   = x_smem + (size_t)BLOCK_N * D_SMEM;
    float* csq_smem = reinterpret_cast<float*>(
                          c_smem + (size_t)PIPE_STAGES * BLOCK_K * D_SMEM);

    const int row_top_in_warp = lane / 4;
    const int row_bot_in_warp = row_top_in_warp + 8;
    const int col_in_atom     = (lane % 4) * 2;

    const int ldm_row_off      = (lane & 8)  ? 8 : 0;
    const int ldm_col_off      = (lane & 16) ? 8 : 0;
    const int ldm_row_in_half  = lane & 7;
    const int ldm_n_atom_off   = (lane & 8) ? 8 : 0;

    // K must fit in one chunk; dispatcher enforces this. Asserts removed for
    // perf — the K<=BLOCK_K invariant holds at the call site.
    const int num_k_chunks = 1;

    // Load x_smem.
    async_load_tile<T, THREADS_PER_CTA>(
        x_smem,
        x + (size_t)pid_b * N * D + (size_t)n_start * D,
        n_count, BLOCK_N, D, D_SMEM);
    ptx::cp_async_commit();

    // Load the (single) means / c_sq tile.
    {
        T* c_dst = c_smem;
        async_load_tile<T, THREADS_PER_CTA>(
            c_dst,
            means + (size_t)pid_b * K * D,
            K, BLOCK_K, D, D_SMEM);
        load_csq_tile<THREADS_PER_CTA>(
            csq_smem, c_sq + (size_t)pid_b * K, K, BLOCK_K);
        ptx::cp_async_commit();
    }
    ptx::cp_async_wait_group<0>();
    __syncthreads();

    // Cache x_sq per row in registers.
    float xs_top_cache[M_ATOMS_PER_WARP];
    float xs_bot_cache[M_ATOMS_PER_WARP];
    bool  top_valid_cache[M_ATOMS_PER_WARP];
    bool  bot_valid_cache[M_ATOMS_PER_WARP];

    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
        int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
        top_valid_cache[m] = (row_top < n_count);
        bot_valid_cache[m] = (row_bot < n_count);
        xs_top_cache[m] = top_valid_cache[m]
            ? x_sq[(size_t)pid_b * N + n_start + row_top] : 0.f;
        xs_bot_cache[m] = bot_valid_cache[m]
            ? x_sq[(size_t)pid_b * N + n_start + row_bot] : 0.f;
    }

    // ------ Single GEMM pass ------
    float acc[M_ATOMS_PER_WARP][N_ATOMS_PER_WARP][4];
    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m)
        #pragma unroll
        for (int n = 0; n < N_ATOMS_PER_WARP; ++n)
            acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;

    #pragma unroll 4
    for (int d_off = 0; d_off < D; d_off += BLOCK_D) {
        uint32_t a_regs[M_ATOMS_PER_WARP][4];
        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            int m_base = warp_id * WARP_M + m * 16;
            int row    = m_base + ldm_row_off + ldm_row_in_half;
            unsigned int smem_addr = ptx::cvta_to_shared(
                x_smem + (size_t)row * D_SMEM + d_off + ldm_col_off);
            ptx::ldmatrix_x4(a_regs[m][0], a_regs[m][1],
                             a_regs[m][2], a_regs[m][3], smem_addr);
        }

        uint32_t b_regs[N_ATOMS_PER_WARP][2];
        #pragma unroll
        for (int n = 0; n < N_ATOMS_PER_WARP; n += 2) {
            int n_col = n * 8 + ldm_n_atom_off + ldm_row_in_half;
            unsigned int smem_addr = ptx::cvta_to_shared(
                c_smem + (size_t)n_col * D_SMEM + d_off + ldm_col_off);
            uint32_t r0, r1, r2, r3;
            ptx::ldmatrix_x4(r0, r1, r2, r3, smem_addr);
            b_regs[n    ][0] = r0;
            b_regs[n + 1][0] = r1;
            b_regs[n    ][1] = r2;
            b_regs[n + 1][1] = r3;
        }

        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
                if constexpr (std::is_same<T, __half>::value ||
                              std::is_same<T, at::Half>::value) {
                    ptx::mma_m16n8k16_fp16(
                        acc[m][n][0], acc[m][n][1],
                        acc[m][n][2], acc[m][n][3],
                        a_regs[m][0], a_regs[m][1],
                        a_regs[m][2], a_regs[m][3],
                        b_regs[n][0], b_regs[n][1],
                        acc[m][n][0], acc[m][n][1],
                        acc[m][n][2], acc[m][n][3]);
                } else {
                    ptx::mma_m16n8k16_bf16(
                        acc[m][n][0], acc[m][n][1],
                        acc[m][n][2], acc[m][n][3],
                        a_regs[m][0], a_regs[m][1],
                        a_regs[m][2], a_regs[m][3],
                        b_regs[n][0], b_regs[n][1],
                        acc[m][n][0], acc[m][n][1],
                        acc[m][n][2], acc[m][n][3]);
                }
            }
        }
    }  // d-loop

    // ------ Pass 1: convert cross-products to logits, build LSE per row.
    // Cache the per-row per-k LOGIT VALUES in registers (overwriting acc[]).
    const float* var_b   = var   + (size_t)pid_b * K;
    const float* log_w_b = log_w + (size_t)pid_b * K;
    const float half_D_log2pi = 0.5f * (float)D * 1.8378770664f;

    LSE lse_top[M_ATOMS_PER_WARP];
    LSE lse_bot[M_ATOMS_PER_WARP];
    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        lse_top[m] = LSE{-FLT_MAX, 0.f};
        lse_bot[m] = LSE{-FLT_MAX, 0.f};
    }

    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        const float xs_top = xs_top_cache[m];
        const float xs_bot = xs_bot_cache[m];
        const bool  tv     = top_valid_cache[m];
        const bool  bv     = bot_valid_cache[m];

        #pragma unroll
        for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
            int k_in_chunk_0 = n * 8 + col_in_atom;
            int k_in_chunk_1 = k_in_chunk_0 + 1;
            int k0 = k_in_chunk_0;
            int k1 = k_in_chunk_1;
            bool k0v = (k0 < K);
            bool k1v = (k1 < K);

            float cross_top0 = acc[m][n][0];
            float cross_top1 = acc[m][n][1];
            float cross_bot0 = acc[m][n][2];
            float cross_bot1 = acc[m][n][3];

            float cs0 = csq_smem[k_in_chunk_0];
            float cs1 = csq_smem[k_in_chunk_1];

            // Convert cross-product to logit, store back into acc[].
            float logit_top0 = -INFINITY, logit_top1 = -INFINITY;
            float logit_bot0 = -INFINITY, logit_bot1 = -INFINITY;
            if (k0v) {
                float v0 = var_b[k0];
                float dist_t = xs_top + cs0 - 2.f * cross_top0;
                float dist_b = xs_bot + cs0 - 2.f * cross_bot0;
                if (dist_t < 0.f) dist_t = 0.f;
                if (dist_b < 0.f) dist_b = 0.f;
                float pre = log_w_b[k0] - half_D_log2pi - 0.5f * (float)D * logf(v0);
                logit_top0 = pre - 0.5f * dist_t / v0;
                logit_bot0 = pre - 0.5f * dist_b / v0;
            }
            if (k1v) {
                float v1 = var_b[k1];
                float dist_t = xs_top + cs1 - 2.f * cross_top1;
                float dist_b = xs_bot + cs1 - 2.f * cross_bot1;
                if (dist_t < 0.f) dist_t = 0.f;
                if (dist_b < 0.f) dist_b = 0.f;
                float pre = log_w_b[k1] - half_D_log2pi - 0.5f * (float)D * logf(v1);
                logit_top1 = pre - 0.5f * dist_t / v1;
                logit_bot1 = pre - 0.5f * dist_b / v1;
            }

            // Cache logit values (overwriting acc registers).
            acc[m][n][0] = logit_top0;
            acc[m][n][1] = logit_top1;
            acc[m][n][2] = logit_bot0;
            acc[m][n][3] = logit_bot1;

            // LSE updates (skip lanes whose row is out-of-range).
            if (tv && k0v) update_lse(lse_top[m], logit_top0);
            if (tv && k1v) update_lse(lse_top[m], logit_top1);
            if (bv && k0v) update_lse(lse_bot[m], logit_bot0);
            if (bv && k1v) update_lse(lse_bot[m], logit_bot1);
        }
    }

    // ------ Reduce (m, s) across the 4 lanes that share a row. ------
    auto warp_reduce_lse = [&](LSE& l) {
        #pragma unroll
        for (int offset : {1, 2}) {
            float other_m = __shfl_xor_sync(0xffffffff, l.m, offset, 4);
            float other_s = __shfl_xor_sync(0xffffffff, l.s, offset, 4);
            float new_m   = fmaxf(l.m, other_m);
            float a = (l.m   == -FLT_MAX) ? 0.f : l.s * __expf(l.m   - new_m);
            float b = (other_m == -FLT_MAX) ? 0.f : other_s * __expf(other_m - new_m);
            l.m = new_m;
            l.s = a + b;
        }
    };

    float ln_top[M_ATOMS_PER_WARP];
    float ln_bot[M_ATOMS_PER_WARP];
    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        warp_reduce_lse(lse_top[m]);
        warp_reduce_lse(lse_bot[m]);
        ln_top[m] = (lse_top[m].s > 0.f)
                    ? (lse_top[m].m + logf(lse_top[m].s)) : -INFINITY;
        ln_bot[m] = (lse_bot[m].s > 0.f)
                    ? (lse_bot[m].m + logf(lse_bot[m].s)) : -INFINITY;
    }

    // ------ Optionally write lse_out (lane%4==0 owns the row). ------
    if (lse_out != nullptr && (lane % 4) == 0) {
        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
            int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
            if (row_top < n_count) {
                lse_out[(size_t)pid_b * N + n_start + row_top] = ln_top[m];
            }
            if (row_bot < n_count) {
                lse_out[(size_t)pid_b * N + n_start + row_bot] = ln_bot[m];
            }
        }
    }

    // ------ Pass 2: write resp[n,k] = exp(logit - lse). ------
    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
        int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
        const float lt = ln_top[m];
        const float lb = ln_bot[m];

        #pragma unroll
        for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
            int k_in_chunk_0 = n * 8 + col_in_atom;
            int k_in_chunk_1 = k_in_chunk_0 + 1;
            int k0 = k_in_chunk_0;
            int k1 = k_in_chunk_1;
            bool k0v = (k0 < K);
            bool k1v = (k1 < K);

            const size_t base_top = ((size_t)pid_b * N + n_start + row_top) * (size_t)K;
            const size_t base_bot = ((size_t)pid_b * N + n_start + row_bot) * (size_t)K;

            if (top_valid_cache[m]) {
                if (k0v) {
                    float r0 = (lt == -INFINITY) ? 0.f : __expf(acc[m][n][0] - lt);
                    resp_out[base_top + k0] = r0;
                }
                if (k1v) {
                    float r1 = (lt == -INFINITY) ? 0.f : __expf(acc[m][n][1] - lt);
                    resp_out[base_top + k1] = r1;
                }
            }
            if (bot_valid_cache[m]) {
                if (k0v) {
                    float r0 = (lb == -INFINITY) ? 0.f : __expf(acc[m][n][2] - lb);
                    resp_out[base_bot + k0] = r0;
                }
                if (k1v) {
                    float r1 = (lb == -INFINITY) ? 0.f : __expf(acc[m][n][3] - lb);
                    resp_out[base_bot + k1] = r1;
                }
            }
        }
    }
}  // logsumexp_resp_sm80_kernel

// ------------------------------------------------------------------
// SMEM size computation
// ------------------------------------------------------------------
static inline size_t smem_bytes_assign(
    int BLOCK_N_, int BLOCK_K_, int D, size_t elt_sz) {
    int D_SMEM = D + SMEM_PAD;
    return (size_t)BLOCK_N_ * D_SMEM * elt_sz         // x_smem
         + (size_t)2 * BLOCK_K_ * D_SMEM * elt_sz     // c_smem (2 stages)
         + (size_t)2 * BLOCK_K_ * sizeof(float);       // csq_smem (2 stages)
}

// ------------------------------------------------------------------
// Typed launcher
// ------------------------------------------------------------------
template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_>
static void launch_assign_sm80_typed(
    const at::Tensor& x,
    const at::Tensor& means,
    const at::Tensor& var,
    const at::Tensor& log_w,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    at::Tensor&       out,
    int B, int N, int K, int D,
    cudaStream_t stream) {

    size_t smem_bytes = smem_bytes_assign(BLOCK_N_, BLOCK_K_, D, sizeof(T));

    auto fn = assign_sm80_kernel<T, BLOCK_N_, BLOCK_K_, WARPS_>;
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    }

    dim3 grid((N + BLOCK_N_ - 1) / BLOCK_N_, B);
    dim3 block(WARPS_ * 32);
    fn<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(means.data_ptr()),
        var.data_ptr<float>(),
        log_w.data_ptr<float>(),
        x_sq.data_ptr<float>(),
        c_sq.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        B, N, K, D);
}

// ------------------------------------------------------------------
// Typed launcher (logsumexp)
// ------------------------------------------------------------------
template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_>
static void launch_logsumexp_sm80_typed(
    const at::Tensor& x,
    const at::Tensor& means,
    const at::Tensor& var,
    const at::Tensor& log_w,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    at::Tensor&       out,
    int B, int N, int K, int D,
    cudaStream_t stream) {

    size_t smem_bytes = smem_bytes_assign(BLOCK_N_, BLOCK_K_, D, sizeof(T));

    auto fn = logsumexp_sm80_kernel<T, BLOCK_N_, BLOCK_K_, WARPS_>;
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    }

    dim3 grid((N + BLOCK_N_ - 1) / BLOCK_N_, B);
    dim3 block(WARPS_ * 32);
    fn<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(means.data_ptr()),
        var.data_ptr<float>(),
        log_w.data_ptr<float>(),
        x_sq.data_ptr<float>(),
        c_sq.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, K, D);
}

// ------------------------------------------------------------------
// Typed launcher (resp)
// ------------------------------------------------------------------
template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_>
static void launch_resp_sm80_typed(
    const at::Tensor& x,
    const at::Tensor& means,
    const at::Tensor& var,
    const at::Tensor& log_w,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    const at::Tensor& log_norm,
    at::Tensor&       out,
    int B, int N, int K, int D,
    cudaStream_t stream) {

    size_t smem_bytes = smem_bytes_assign(BLOCK_N_, BLOCK_K_, D, sizeof(T));

    auto fn = resp_sm80_kernel<T, BLOCK_N_, BLOCK_K_, WARPS_>;
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    }

    dim3 grid((N + BLOCK_N_ - 1) / BLOCK_N_, B);
    dim3 block(WARPS_ * 32);
    fn<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(means.data_ptr()),
        var.data_ptr<float>(),
        log_w.data_ptr<float>(),
        x_sq.data_ptr<float>(),
        c_sq.data_ptr<float>(),
        log_norm.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, K, D);
}

// ------------------------------------------------------------------
// Typed launcher (fused logsumexp+resp)
// ------------------------------------------------------------------
template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_>
static void launch_logsumexp_resp_sm80_typed(
    const at::Tensor& x,
    const at::Tensor& means,
    const at::Tensor& var,
    const at::Tensor& log_w,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    at::Tensor*       lse_out,        // optional
    at::Tensor&       resp_out,
    int B, int N, int K, int D,
    cudaStream_t stream) {

    size_t smem_bytes = smem_bytes_assign(BLOCK_N_, BLOCK_K_, D, sizeof(T));

    auto fn = logsumexp_resp_sm80_kernel<T, BLOCK_N_, BLOCK_K_, WARPS_>;
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    }

    dim3 grid((N + BLOCK_N_ - 1) / BLOCK_N_, B);
    dim3 block(WARPS_ * 32);
    fn<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(means.data_ptr()),
        var.data_ptr<float>(),
        log_w.data_ptr<float>(),
        x_sq.data_ptr<float>(),
        c_sq.data_ptr<float>(),
        lse_out ? lse_out->data_ptr<float>() : nullptr,
        resp_out.data_ptr<float>(),
        B, N, K, D);
}

// ------------------------------------------------------------------
// Input validation helper
// ------------------------------------------------------------------
static void _check_sm80_inputs(
    const at::Tensor& x, const at::Tensor& means,
    const at::Tensor& var, const at::Tensor& log_w,
    const at::Tensor& x_sq, const at::Tensor& c_sq) {
    TORCH_CHECK(x.is_cuda(), "x must be on a CUDA device");
    TORCH_CHECK(x.is_contiguous() && means.is_contiguous() &&
                var.is_contiguous() && log_w.is_contiguous() &&
                x_sq.is_contiguous() && c_sq.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(means.scalar_type() == x.scalar_type(),
                "means must match x dtype");
    TORCH_CHECK(var.scalar_type() == at::kFloat &&
                log_w.scalar_type() == at::kFloat,
                "var and log_w must be float32");
    TORCH_CHECK(x_sq.scalar_type() == at::kFloat &&
                c_sq.scalar_type() == at::kFloat,
                "x_sq and c_sq must be float32");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "assign_sm80 requires fp16 or bf16 input (got ", x.scalar_type(), ")");
    TORCH_CHECK(x.dim() == 3 && means.dim() == 3,
                "x must be (B,N,D); means must be (B,K,D)");
    int B = (int)x.size(0), N = (int)x.size(1), D = (int)x.size(2);
    int K = (int)means.size(1);
    TORCH_CHECK(means.size(0) == B && means.size(2) == D,
                "means must be (B,K,D) matching x");
    TORCH_CHECK(var.dim() == 2 && var.size(0) == B && (int)var.size(1) == K,
                "var must be (B,K)");
    TORCH_CHECK(log_w.sizes() == var.sizes(), "log_w must match var shape");
    TORCH_CHECK(x_sq.dim() == 2 && x_sq.size(0) == B && x_sq.size(1) == N,
                "x_sq must be (B,N)");
    TORCH_CHECK(c_sq.dim() == 2 && c_sq.size(0) == B && (int)c_sq.size(1) == K,
                "c_sq must be (B,K)");
    TORCH_CHECK(D % BLOCK_D == 0,
                "assign_sm80: D must be a multiple of 16 (got ", D, ")");
}

}  // anonymous namespace

// ------------------------------------------------------------------
// Public API: assign_sm80
// ------------------------------------------------------------------
at::Tensor assign_sm80(
    const at::Tensor& x,
    const at::Tensor& means,
    const at::Tensor& var,
    const at::Tensor& log_w,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    c10::optional<at::Tensor> out) {

    _check_sm80_inputs(x, means, var, log_w, x_sq, c_sq);
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);

    at::Tensor result;
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == at::kInt &&
                    out->sizes() == at::IntArrayRef({B, N}),
                    "out must be int32 (B,N)");
        result = *out;
    } else {
        result = at::empty({B, N}, x.options().dtype(at::kInt));
    }

    if (N == 0) return result;

    // Check device SMEM limit to pick tile configuration.
    int dev = x.device().index();
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, dev);
    size_t smem_limit = props.sharedMemPerBlockOptin;
    if (smem_limit == 0) smem_limit = props.sharedMemPerBlock;

    auto try_launch = [&](auto tag, int bn, int bk, int warps) -> bool {
        using T = decltype(tag);
        size_t smem = smem_bytes_assign(bn, bk, D, sizeof(T));
        if (smem > smem_limit) return false;
        // Only instantiate the compile-time combinations we need.
        // Wide tile: BLOCK_N=128, BLOCK_K=64, WARPS=4
        if (bn == 128 && bk == 64 && warps == 4) {
            launch_assign_sm80_typed<T, 128, 64, 4>(
                x, means, var, log_w, x_sq, c_sq, result, B, N, K, D, stream);
            return true;
        }
        // Exp59: BLOCK_N=128, BLOCK_K=32, WARPS=4. Halves c_smem vs (128,64,4)
        // when K<=32, raising max-CTAs-per-SM and shrinking the per-CTA SMEM
        // footprint. For K=32 shapes the BK=64 tile wastes half its SMEM.
        if (bn == 128 && bk == 32 && warps == 4) {
            launch_assign_sm80_typed<T, 128, 32, 4>(
                x, means, var, log_w, x_sq, c_sq, result, B, N, K, D, stream);
            return true;
        }
        // Narrow tile: BLOCK_N=64, BLOCK_K=32, WARPS=4 (smaller SMEM footprint)
        if (bn == 64 && bk == 32 && warps == 4) {
            launch_assign_sm80_typed<T, 64, 32, 4>(
                x, means, var, log_w, x_sq, c_sq, result, B, N, K, D, stream);
            return true;
        }
        return false;
    };

    // Exp59: prefer (128, 32, 4) when K is small enough that BK=64 would be
    // half-empty. Falls through to the legacy ladder otherwise.
    bool launched = false;
    if (x.scalar_type() == at::kHalf) {
        if (K <= 32) {
            launched = try_launch(__half{}, 128, 32, 4);
        }
        if (!launched) {
            launched = try_launch(__half{}, 128, 64, 4) ||
                       try_launch(__half{}, 128, 32, 4) ||
                       try_launch(__half{}, 64,  32, 4);
        }
    } else {
        if (K <= 32) {
            launched = try_launch(__nv_bfloat16{}, 128, 32, 4);
        }
        if (!launched) {
            launched = try_launch(__nv_bfloat16{}, 128, 64, 4) ||
                       try_launch(__nv_bfloat16{}, 128, 32, 4) ||
                       try_launch(__nv_bfloat16{}, 64,  32, 4);
        }
    }

    if (!launched) {
        // No tile fits in SMEM — fall back to the safe scalar kernel.
        return assign_safe(x, means, var, log_w, std::move(out));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

// ------------------------------------------------------------------
// Public API: logsumexp_sm80 (Plan 4 Task 5)
// ------------------------------------------------------------------
// Same dispatch shape as assign_sm80: try wide tile (BLOCK_N=128, BLOCK_K=64,
// WARPS=4), fall back to narrow tile (BLOCK_N=64, BLOCK_K=32, WARPS=4) if SMEM
// is tight, then fall back to logsumexp_safe if neither tile fits.
at::Tensor logsumexp_sm80(
    const at::Tensor& x,
    const at::Tensor& means,
    const at::Tensor& var,
    const at::Tensor& log_w,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    c10::optional<at::Tensor> out) {

    _check_sm80_inputs(x, means, var, log_w, x_sq, c_sq);
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);

    at::Tensor result;
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == at::kFloat &&
                    out->sizes() == at::IntArrayRef({B, N}),
                    "out must be float32 (B,N)");
        result = *out;
    } else {
        result = at::empty({B, N}, x.options().dtype(at::kFloat));
    }

    if (N == 0) return result;

    int dev = x.device().index();
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, dev);
    size_t smem_limit = props.sharedMemPerBlockOptin;
    if (smem_limit == 0) smem_limit = props.sharedMemPerBlock;

    auto try_launch = [&](auto tag, int bn, int bk, int warps) -> bool {
        using T = decltype(tag);
        size_t smem = smem_bytes_assign(bn, bk, D, sizeof(T));
        if (smem > smem_limit) return false;
        if (bn == 128 && bk == 64 && warps == 4) {
            launch_logsumexp_sm80_typed<T, 128, 64, 4>(
                x, means, var, log_w, x_sq, c_sq, result, B, N, K, D, stream);
            return true;
        }
        // Exp59: small-K-friendly tile (see assign_sm80 dispatch comment).
        if (bn == 128 && bk == 32 && warps == 4) {
            launch_logsumexp_sm80_typed<T, 128, 32, 4>(
                x, means, var, log_w, x_sq, c_sq, result, B, N, K, D, stream);
            return true;
        }
        if (bn == 64 && bk == 32 && warps == 4) {
            launch_logsumexp_sm80_typed<T, 64, 32, 4>(
                x, means, var, log_w, x_sq, c_sq, result, B, N, K, D, stream);
            return true;
        }
        return false;
    };

    bool launched = false;
    if (x.scalar_type() == at::kHalf) {
        if (K <= 32) {
            launched = try_launch(__half{}, 128, 32, 4);
        }
        if (!launched) {
            launched = try_launch(__half{}, 128, 64, 4) ||
                       try_launch(__half{}, 128, 32, 4) ||
                       try_launch(__half{}, 64,  32, 4);
        }
    } else {
        if (K <= 32) {
            launched = try_launch(__nv_bfloat16{}, 128, 32, 4);
        }
        if (!launched) {
            launched = try_launch(__nv_bfloat16{}, 128, 64, 4) ||
                       try_launch(__nv_bfloat16{}, 128, 32, 4) ||
                       try_launch(__nv_bfloat16{}, 64,  32, 4);
        }
    }

    if (!launched) {
        return logsumexp_safe(x, means, var, log_w, std::move(out));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

// ------------------------------------------------------------------
// Public API: resp_sm80 (Plan 4 Task 6 — fully implemented)
// ------------------------------------------------------------------
// Same tile dispatch as assign_sm80 / logsumexp_sm80.
// Falls back to resp_safe if no tile fits in SMEM.
at::Tensor resp_sm80(
    const at::Tensor& x,
    const at::Tensor& means,
    const at::Tensor& var,
    const at::Tensor& log_w,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    const at::Tensor& log_norm,
    c10::optional<at::Tensor> out) {

    _check_sm80_inputs(x, means, var, log_w, x_sq, c_sq);
    TORCH_CHECK(log_norm.is_cuda() && log_norm.is_contiguous() &&
                log_norm.scalar_type() == at::kFloat,
                "log_norm must be contiguous fp32 on CUDA");
    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);
    TORCH_CHECK(log_norm.sizes() == at::IntArrayRef({B, N}),
                "log_norm must be (B, N)");

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    at::Tensor result;
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == at::kFloat &&
                    out->sizes() == at::IntArrayRef({B, N, K}),
                    "out must be float32 (B,N,K)");
        result = *out;
    } else {
        result = at::empty({B, N, K}, x.options().dtype(at::kFloat));
    }

    if (N == 0) return result;

    int dev = x.device().index();
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, dev);
    size_t smem_limit = props.sharedMemPerBlockOptin;
    if (smem_limit == 0) smem_limit = props.sharedMemPerBlock;

    auto try_launch = [&](auto tag, int bn, int bk, int warps) -> bool {
        using T = decltype(tag);
        size_t smem = smem_bytes_assign(bn, bk, D, sizeof(T));
        if (smem > smem_limit) return false;
        if (bn == 128 && bk == 64 && warps == 4) {
            launch_resp_sm80_typed<T, 128, 64, 4>(
                x, means, var, log_w, x_sq, c_sq, log_norm, result,
                B, N, K, D, stream);
            return true;
        }
        // Exp59: small-K-friendly tile (see assign_sm80 dispatch comment).
        if (bn == 128 && bk == 32 && warps == 4) {
            launch_resp_sm80_typed<T, 128, 32, 4>(
                x, means, var, log_w, x_sq, c_sq, log_norm, result,
                B, N, K, D, stream);
            return true;
        }
        if (bn == 64 && bk == 32 && warps == 4) {
            launch_resp_sm80_typed<T, 64, 32, 4>(
                x, means, var, log_w, x_sq, c_sq, log_norm, result,
                B, N, K, D, stream);
            return true;
        }
        return false;
    };

    bool launched = false;
    if (x.scalar_type() == at::kHalf) {
        if (K <= 32) {
            launched = try_launch(__half{}, 128, 32, 4);
        }
        if (!launched) {
            launched = try_launch(__half{}, 128, 64, 4) ||
                       try_launch(__half{}, 128, 32, 4) ||
                       try_launch(__half{}, 64,  32, 4);
        }
    } else {
        if (K <= 32) {
            launched = try_launch(__nv_bfloat16{}, 128, 32, 4);
        }
        if (!launched) {
            launched = try_launch(__nv_bfloat16{}, 128, 64, 4) ||
                       try_launch(__nv_bfloat16{}, 128, 32, 4) ||
                       try_launch(__nv_bfloat16{}, 64,  32, 4);
        }
    }

    if (!launched) {
        return resp_safe(x, means, var, log_w, log_norm, std::move(out));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

// ------------------------------------------------------------------
// Public API: logsumexp_resp_sm80 (Exp62 — fused E-step kernel)
//
// Returns (lse, resp) in one kernel launch. Constraint: K must be <=
// BLOCK_K of the chosen tile (the kernel only does one K-chunk).
// Returns std::nullopt if no tile fits — caller must fall back.
// ------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor> logsumexp_resp_sm80(
    const at::Tensor& x,
    const at::Tensor& means,
    const at::Tensor& var,
    const at::Tensor& log_w,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    bool need_lse) {

    _check_sm80_inputs(x, means, var, log_w, x_sq, c_sq);
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)means.size(1);

    auto resp = at::empty({B, N, K}, x.options().dtype(at::kFloat));
    at::Tensor lse;
    at::Tensor* lse_ptr = nullptr;
    if (need_lse) {
        lse = at::empty({B, N}, x.options().dtype(at::kFloat));
        lse_ptr = &lse;
    } else {
        lse = at::empty({0}, x.options().dtype(at::kFloat));
    }

    if (N == 0) return std::make_tuple(lse, resp);

    int dev = x.device().index();
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, dev);
    size_t smem_limit = props.sharedMemPerBlockOptin;
    if (smem_limit == 0) smem_limit = props.sharedMemPerBlock;

    auto try_launch = [&](auto tag, int bn, int bk, int warps) -> bool {
        using T = decltype(tag);
        // Constraint: kernel only handles K <= BLOCK_K (single K-chunk).
        if (K > bk) return false;
        size_t smem = smem_bytes_assign(bn, bk, D, sizeof(T));
        if (smem > smem_limit) return false;
        if (bn == 128 && bk == 64 && warps == 4) {
            launch_logsumexp_resp_sm80_typed<T, 128, 64, 4>(
                x, means, var, log_w, x_sq, c_sq, lse_ptr, resp,
                B, N, K, D, stream);
            return true;
        }
        if (bn == 128 && bk == 32 && warps == 4) {
            launch_logsumexp_resp_sm80_typed<T, 128, 32, 4>(
                x, means, var, log_w, x_sq, c_sq, lse_ptr, resp,
                B, N, K, D, stream);
            return true;
        }
        if (bn == 64 && bk == 32 && warps == 4) {
            launch_logsumexp_resp_sm80_typed<T, 64, 32, 4>(
                x, means, var, log_w, x_sq, c_sq, lse_ptr, resp,
                B, N, K, D, stream);
            return true;
        }
        return false;
    };

    bool launched = false;
    if (x.scalar_type() == at::kHalf) {
        if (K <= 32) launched = try_launch(__half{}, 128, 32, 4);
        if (!launched && K <= 64) launched = try_launch(__half{}, 128, 64, 4);
        if (!launched && K <= 32) launched = try_launch(__half{}, 64, 32, 4);
    } else {
        if (K <= 32) launched = try_launch(__nv_bfloat16{}, 128, 32, 4);
        if (!launched && K <= 64) launched = try_launch(__nv_bfloat16{}, 128, 64, 4);
        if (!launched && K <= 32) launched = try_launch(__nv_bfloat16{}, 64, 32, 4);
    }

    if (!launched) {
        // Caller must fall back. Return empty tensors as a sentinel.
        return std::make_tuple(at::empty({0}, x.options().dtype(at::kFloat)),
                               at::empty({0}, x.options().dtype(at::kFloat)));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(lse, resp);
}

}}}  // namespace gmmxx::estep::spherical
