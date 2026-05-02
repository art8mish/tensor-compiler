#pragma once

#include <cstddef>

namespace tensor_compiler::kernels {

void matmul_naive(const float *A, const float *B, float *C, std::size_t M, std::size_t K,
                  std::size_t N);

// i–k–j loop order
void matmul_reordered(const float *A, const float *B, float *C, std::size_t M, std::size_t K,
                      std::size_t N);

// Blocked (tiled)
void matmul_tiled(const float *A, const float *B, float *C, std::size_t M, std::size_t K,
                  std::size_t N);

// Tiling + AVX2/FMA
void matmul_optimized(const float *A, const float *B, float *C, std::size_t M, std::size_t K,
                      std::size_t N);

} // namespace tensor_compiler::kernels
