#include "kernels/conv.hpp"
#include "kernels/matmul.hpp"
#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

namespace {

using tensor_compiler::kernels::conv2d_im2col;
using tensor_compiler::kernels::conv2d_naive;
using tensor_compiler::kernels::matmul_naive;
using tensor_compiler::kernels::matmul_optimized;
using tensor_compiler::kernels::matmul_reordered;
using tensor_compiler::kernels::matmul_tiled;

struct MatmulData {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
};

MatmulData gen_matmul_data(std::size_t n, float a_value = 1.0f, float b_value = 1.0f) {
    MatmulData data{
        std::vector<float>(n * n, a_value),
        std::vector<float>(n * n, b_value),
        std::vector<float>(n * n),
    };
    return data;
}

template <typename Fn> void bench_matmul(benchmark::State &state, Fn fn) {
    const auto n = static_cast<std::size_t>(state.range(0));
    MatmulData data = gen_matmul_data(n);
    for (auto _ : state)
        fn(data.a.data(), data.b.data(), data.c.data(), n, n, n);
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(n * n * n * 2));
}

void bench_matmul_naive(benchmark::State &state) {
    bench_matmul(state, matmul_naive);
}

void bench_matmul_reordered(benchmark::State &state) {
    bench_matmul(state, matmul_reordered);
}

void bench_matmul_tiled(benchmark::State &state) {
    bench_matmul(state, matmul_tiled);
}

void bench_matmul_optimized(benchmark::State &state) {
    bench_matmul(state, matmul_optimized);
}

struct ConvData {
    std::vector<float> in;
    std::vector<float> w;
    std::vector<float> out;
};

ConvData gen_conv_data(std::int64_t C_in, std::int64_t H, std::int64_t C_out, std::int64_t K,
                       float in_value = 0.3f, float w_value = 0.1f) {
    const std::int64_t W = H;
    const std::int64_t OH = H - K + 1;
    const std::int64_t OW = W - K + 1;
    ConvData data{
        std::vector<float>(static_cast<size_t>(C_in * H * W), in_value),
        std::vector<float>(static_cast<size_t>(C_out * C_in * K * K), w_value),
        std::vector<float>(static_cast<size_t>(C_out * OH * OW)),
    };
    return data;
}

template <typename Fn> void bench_conv(benchmark::State &state, Fn fn) {
    const std::int64_t C_in = state.range(0);
    const std::int64_t H = state.range(1);
    const std::int64_t C_out = 8;
    const std::int64_t K = 3;
    const std::int64_t W = H;
    ConvData data = gen_conv_data(C_in, H, C_out, K);

    for (auto _ : state)
        fn(data.in.data(), data.w.data(), data.out.data(), C_in, H, W, C_out, K, K, 1, 1, 0, 0);
}

void bench_conv_naive(benchmark::State &state) {
    bench_conv(state, conv2d_naive);
}

void bench_conv_im2col(benchmark::State &state) {
    bench_conv(state, conv2d_im2col);
}

} // namespace

BENCHMARK(bench_matmul_naive)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(384);
BENCHMARK(bench_matmul_reordered)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(384);
BENCHMARK(bench_matmul_tiled)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(384);
BENCHMARK(bench_matmul_optimized)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(384);

BENCHMARK(bench_conv_naive)->Args({8, 32})->Args({16, 64})->Args({32, 128});
BENCHMARK(bench_conv_im2col)->Args({8, 32})->Args({16, 64})->Args({32, 128});

BENCHMARK_MAIN();
