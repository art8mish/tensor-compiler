#include <kernels/conv.hpp>
#include <kernels/matmul.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using tensor_compiler::kernels::conv2d_im2col;
using tensor_compiler::kernels::conv2d_naive;
using tensor_compiler::kernels::matmul_naive;
using tensor_compiler::kernels::matmul_optimized;
using tensor_compiler::kernels::matmul_reordered;
using tensor_compiler::kernels::matmul_tiled;

namespace {

void expect_matmul_close(const std::vector<float> &ref, const std::vector<float> &got,
                         std::size_t elems, float eps = 1e-3f) {
    ASSERT_EQ(ref.size(), got.size());
    for (std::size_t i = 0; i < elems; ++i)
        EXPECT_NEAR(ref[i], got[i], eps) << " at i=" << i;
}

} // namespace

TEST(KernelsMatmul, MatchesNaiveSmall) {
    const std::size_t M = 7, K = 11, N = 5;
    std::vector<float> a(M * K), b(K * N), r0(M * N), r1(M * N), r2(M * N), r3(M * N);
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<float>(i % 13) * 0.1f - 0.5f;
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<float>(i % 7) * 0.15f - 0.3f;

    matmul_naive(a.data(), b.data(), r0.data(), M, K, N);
    matmul_reordered(a.data(), b.data(), r1.data(), M, K, N);
    matmul_tiled(a.data(), b.data(), r2.data(), M, K, N);
    matmul_optimized(a.data(), b.data(), r3.data(), M, K, N);

    expect_matmul_close(r0, r1, M * N);
    expect_matmul_close(r0, r2, M * N);
    expect_matmul_close(r0, r3, M * N);
}

TEST(KernelsMatmul, MatchesNaiveSquarePowerOfTwo) {
    const std::size_t n = 128;
    std::vector<float> a(n * n), b(n * n), r0(n * n), r3(n * n);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = std::sin(static_cast<float>(i)) * 0.01f;
        b[i] = std::cos(static_cast<float>(i)) * 0.01f;
    }
    matmul_naive(a.data(), b.data(), r0.data(), n, n, n);
    matmul_optimized(a.data(), b.data(), r3.data(), n, n, n);
    expect_matmul_close(r0, r3, n * n, 5e-3f);
}

TEST(KernelsConv, Im2ColMatchesNaiveValid) {
    const std::int64_t C_in = 3;
    const std::int64_t H = 12;
    const std::int64_t W = 14;
    const std::int64_t C_out = 4;
    const std::int64_t KH = 3;
    const std::int64_t KW = 3;

    std::vector<float> in(static_cast<size_t>(C_in * H * W));
    std::vector<float> w(static_cast<size_t>(C_out * C_in * KH * KW));
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = static_cast<float>(i % 17) * 0.05f - 0.4f;
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = static_cast<float>(i % 11) * 0.07f - 0.2f;

    const std::int64_t OH = H - KH + 1;
    const std::int64_t OW = W - KW + 1;
    std::vector<float> y0(static_cast<size_t>(C_out * OH * OW));
    std::vector<float> y1(static_cast<size_t>(C_out * OH * OW));

    conv2d_naive(in.data(), w.data(), y0.data(), C_in, H, W, C_out, KH, KW, 1, 1, 0, 0);
    conv2d_im2col(in.data(), w.data(), y1.data(), C_in, H, W, C_out, KH, KW, 1, 1, 0, 0);

    for (size_t i = 0; i < y0.size(); ++i)
        EXPECT_NEAR(y0[i], y1[i], 1e-3f) << " i=" << i;
}

TEST(KernelsConv, Im2ColStridePadding) {
    const std::int64_t C_in = 2;
    const std::int64_t H = 10;
    const std::int64_t W = 10;
    const std::int64_t C_out = 3;
    const std::int64_t KH = 3;
    const std::int64_t KW = 3;
    const std::int64_t stride_h = 2;
    const std::int64_t stride_w = 2;
    const std::int64_t pad_h = 1;
    const std::int64_t pad_w = 1;

    std::vector<float> in(static_cast<size_t>(C_in * H * W), 1.0F);
    std::vector<float> w(static_cast<size_t>(C_out * C_in * KH * KW));
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = static_cast<float>(i + 1) * 0.01f;

    const std::int64_t OH = (H + 2 * pad_h - KH) / stride_h + 1;
    const std::int64_t OW = (W + 2 * pad_w - KW) / stride_w + 1;
    std::vector<float> y0(static_cast<size_t>(C_out * OH * OW));
    std::vector<float> y1(static_cast<size_t>(C_out * OH * OW));

    conv2d_naive(in.data(), w.data(), y0.data(), C_in, H, W, C_out, KH, KW, stride_h, stride_w,
                 pad_h, pad_w);
    conv2d_im2col(in.data(), w.data(), y1.data(), C_in, H, W, C_out, KH, KW, stride_h, stride_w,
                  pad_h, pad_w);

    for (size_t i = 0; i < y0.size(); ++i)
        EXPECT_NEAR(y0[i], y1[i], 1e-3f) << " i=" << i;
}
