#pragma once

#include <cstddef>
#include <cstdint>

namespace tensor_compiler::kernels {

/// NCHW: input [C_in, H, W], weight [C_out, C_in, KH, KW], output [C_out, OH, OW].
void conv2d_naive(const float *input, const float *weight, float *output, std::int64_t C_in,
                  std::int64_t H, std::int64_t W, std::int64_t C_out, std::int64_t KH,
                  std::int64_t KW, std::int64_t stride_h, std::int64_t stride_w, std::int64_t pad_h,
                  std::int64_t pad_w);

void conv2d_im2col(const float *input, const float *weight, float *output, std::int64_t C_in,
                   std::int64_t H, std::int64_t W, std::int64_t C_out, std::int64_t KH,
                   std::int64_t KW, std::int64_t stride_h, std::int64_t stride_w,
                   std::int64_t pad_h, std::int64_t pad_w);

} // namespace tensor_compiler::kernels
