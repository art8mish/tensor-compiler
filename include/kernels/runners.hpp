#pragma once

#include "nodes/operations.hpp"
#include "tensor/tensor.hpp"
#include <cstdint>

namespace tensor_compiler::kernels {

void run_matmul(const Tensor &A, const Tensor &B, Tensor &Y);

void run_conv2d(const Tensor &X, const Tensor &W, Tensor &Y, std::int64_t stride_h = 1,
                std::int64_t stride_w = 1, std::int64_t pad_h = 0, std::int64_t pad_w = 0);

void run_matmul_node(const MatMulNode &node, Tensor &Y);

void run_conv_node(const ConvNode &node, Tensor &Y);

} // namespace tensor_compiler::kernels
