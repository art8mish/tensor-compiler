#include "kernels/matmul.hpp"

#include <algorithm>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace tensor_compiler::kernels {

namespace {

constexpr std::size_t TileM = 64;
constexpr std::size_t TileN = 64;
constexpr std::size_t TileK = 64;

inline void fmadd_row_scalar(std::size_t N, float aik, const float *bk_row, float *ci_row) {
    for (std::size_t j = 0; j < N; ++j)
        ci_row[j] += aik * bk_row[j];
}

#if defined(__AVX2__) && defined(__FMA__)
inline void fmadd_row_avx(std::size_t N, float aik, const float *bk_row, float *ci_row) {
    std::size_t j = 0;
    for (; j + 8 <= N; j += 8) {
        __m256 c = _mm256_loadu_ps(ci_row + j);
        __m256 b = _mm256_loadu_ps(bk_row + j);
        __m256 a = _mm256_set1_ps(aik);
        c = _mm256_fmadd_ps(a, b, c);
        _mm256_storeu_ps(ci_row + j, c);
    }
    for (; j < N; ++j)
        ci_row[j] += aik * bk_row[j];
}
#endif

void matmul_tiled_impl(const float *A, const float *B, float *C, std::size_t M, std::size_t K,
                       std::size_t N,
                       void (*func)(std::size_t N, float aik, const float *bk_row, float *ci_row)) {
    std::memset(C, 0, M * N * sizeof(float));
    for (std::size_t ii = 0; ii < M; ii += TileM) {
        std::size_t i_max = std::min(ii + TileM, M);
        for (std::size_t jj = 0; jj < N; jj += TileN) {
            std::size_t j_max = std::min(jj + TileN, N);
            std::size_t n_tile = j_max - jj;
            for (std::size_t kk = 0; kk < K; kk += TileK) {
                std::size_t k_max = std::min(kk + TileK, K);
                for (std::size_t i = ii; i < i_max; ++i) {
                    for (std::size_t k = kk; k < k_max; ++k) {
                        float aik = A[i * K + k];
                        const float *bk = B + k * N + jj;
                        float *ci = C + i * N + jj;
                        func(n_tile, aik, bk, ci);
                    }
                }
            }
        }
    }
}

} // namespace

void matmul_naive(const float *A, const float *B, float *C, std::size_t M, std::size_t K,
                  std::size_t N) {
    std::memset(C, 0, M * N * sizeof(float));
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

void matmul_reordered(const float *A, const float *B, float *C, std::size_t M, std::size_t K,
                      std::size_t N) {
    std::memset(C, 0, M * N * sizeof(float));
    for (std::size_t i = 0; i < M; ++i) {
        const float *ai = A + i * K;
        float *ci = C + i * N;
        for (std::size_t k = 0; k < K; ++k) {
            float aik = ai[k];
            const float *bk = B + k * N;
            for (std::size_t j = 0; j < N; ++j)
                ci[j] += aik * bk[j];
        }
    }
}

void matmul_tiled(const float *A, const float *B, float *C, std::size_t M, std::size_t K,
                  std::size_t N) {
    matmul_tiled_impl(A, B, C, M, K, N, fmadd_row_scalar);
}

void matmul_optimized(const float *A, const float *B, float *C, std::size_t M, std::size_t K,
                      std::size_t N) {
#if defined(__AVX2__) && defined(__FMA__)
    matmul_tiled_impl(A, B, C, M, K, N, fmadd_row_avx);
#else
    matmul_tiled(A, B, C, M, K, N);
#endif
}

} // namespace tensor_compiler::kernels
