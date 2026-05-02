#include "kernels/conv.hpp"
#include "kernels/matmul.hpp"

#include <cstring>
#include <vector>

namespace tensor_compiler::kernels {

namespace {

inline std::int64_t input_index(std::int64_t C_in, std::int64_t H, std::int64_t W, std::int64_t ic,
                                std::int64_t y, std::int64_t x) {
    return ic * (H * W) + y * W + x;
}

inline std::int64_t weight_index(std::int64_t C_in, std::int64_t KH, std::int64_t KW,
                                 std::int64_t oc, std::int64_t ic, std::int64_t ky,
                                 std::int64_t kx) {
    return ((oc * C_in + ic) * KH + ky) * KW + kx;
}

inline std::int64_t output_index(std::int64_t OH, std::int64_t OW, std::int64_t oc, std::int64_t oy,
                                 std::int64_t ox) {
    return oc * (OH * OW) + oy * OW + ox;
}

} // namespace

void conv2d_naive(const float *input, const float *weight, float *output, std::int64_t C_in,
                  std::int64_t H, std::int64_t W, std::int64_t C_out, std::int64_t KH,
                  std::int64_t KW, std::int64_t stride_h, std::int64_t stride_w, std::int64_t pad_h,
                  std::int64_t pad_w) {
    const std::int64_t OH = (H + 2 * pad_h - KH) / stride_h + 1;
    const std::int64_t OW = (W + 2 * pad_w - KW) / stride_w + 1;

    std::memset(output, 0, static_cast<size_t>(C_out * OH * OW) * sizeof(float));

    for (std::int64_t oc = 0; oc < C_out; ++oc) {
        for (std::int64_t oy = 0; oy < OH; ++oy) {
            for (std::int64_t ox = 0; ox < OW; ++ox) {
                float acc = 0.0F;
                for (std::int64_t ic = 0; ic < C_in; ++ic) {
                    for (std::int64_t ky = 0; ky < KH; ++ky) {
                        for (std::int64_t kx = 0; kx < KW; ++kx) {
                            const std::int64_t iy = oy * stride_h + ky - pad_h;
                            const std::int64_t ix = ox * stride_w + kx - pad_w;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                acc += input[input_index(C_in, H, W, ic, iy, ix)] *
                                       weight[weight_index(C_in, KH, KW, oc, ic, ky, kx)];
                            }
                        }
                    }
                }
                output[output_index(OH, OW, oc, oy, ox)] = acc;
            }
        }
    }
}

void conv2d_im2col(const float *input, const float *weight, float *output, std::int64_t C_in,
                   std::int64_t H, std::int64_t W, std::int64_t C_out, std::int64_t KH,
                   std::int64_t KW, std::int64_t stride_h, std::int64_t stride_w,
                   std::int64_t pad_h, std::int64_t pad_w) {
    const std::int64_t OH = (H + 2 * pad_h - KH) / stride_h + 1;
    const std::int64_t OW = (W + 2 * pad_w - KW) / stride_w + 1;
    const std::int64_t S = OH * OW;
    const std::int64_t P = C_in * KH * KW;

    std::vector<float> im2col(static_cast<size_t>(S * P));

    for (std::int64_t oy = 0; oy < OH; ++oy) {
        for (std::int64_t ox = 0; ox < OW; ++ox) {
            const std::int64_t s = oy * OW + ox;
            float *row = im2col.data() + s * static_cast<std::ptrdiff_t>(P);
            std::int64_t col = 0;
            for (std::int64_t ic = 0; ic < C_in; ++ic) {
                for (std::int64_t ky = 0; ky < KH; ++ky) {
                    for (std::int64_t kx = 0; kx < KW; ++kx, ++col) {
                        const std::int64_t iy = oy * stride_h + ky - pad_h;
                        const std::int64_t ix = ox * stride_w + kx - pad_w;
                        row[col] = (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                       ? input[input_index(C_in, H, W, ic, iy, ix)]
                                       : 0.0F;
                    }
                }
            }
        }
    }

    std::vector<float> b_matrix(static_cast<size_t>(P * C_out));
    for (std::int64_t oc = 0; oc < C_out; ++oc) {
        for (std::int64_t ic = 0; ic < C_in; ++ic) {
            for (std::int64_t ky = 0; ky < KH; ++ky) {
                for (std::int64_t kx = 0; kx < KW; ++kx) {
                    const std::int64_t p = ((ic * KH) + ky) * KW + kx;
                    b_matrix[static_cast<size_t>(p) * static_cast<size_t>(C_out) +
                             static_cast<size_t>(oc)] =
                        weight[weight_index(C_in, KH, KW, oc, ic, ky, kx)];
                }
            }
        }
    }

    std::vector<float> gemm_out(static_cast<size_t>(S * C_out));
    matmul_optimized(im2col.data(), b_matrix.data(), gemm_out.data(), static_cast<std::size_t>(S),
                     static_cast<std::size_t>(P), static_cast<std::size_t>(C_out));

    for (std::int64_t oc = 0; oc < C_out; ++oc) {
        for (std::int64_t oy = 0; oy < OH; ++oy) {
            for (std::int64_t ox = 0; ox < OW; ++ox) {
                const std::int64_t s = oy * OW + ox;
                output[output_index(OH, OW, oc, oy, ox)] =
                    gemm_out[static_cast<size_t>(s) * static_cast<size_t>(C_out) +
                             static_cast<size_t>(oc)];
            }
        }
    }
}

} // namespace tensor_compiler::kernels
