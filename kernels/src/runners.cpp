#include "kernels/runners.hpp"

#include "kernels/conv.hpp"
#include "kernels/matmul.hpp"
#include "nodes/operations.hpp"
#include "nodes/tensor_node.hpp"
#include "tensor/tensor.hpp"

#include <stdexcept>
#include <vector>

namespace tensor_compiler::kernels {

namespace {

void check_float32(const Tensor &t, const char *name) {
    if (t.dtype() != DataType::FLOAT32)
        throw std::invalid_argument(std::string(name) + " must have float32 dtype");
}

void check_rank(const Tensor &t, size_t rank, const char *name) {
    if (t.shape().size() != rank)
        throw std::invalid_argument(std::string(name) + " must be rank " + std::to_string(rank));
}

} // namespace

void run_matmul(const Tensor &A, const Tensor &B, Tensor &Y) {
    check_float32(A, "A");
    check_float32(B, "B");
    check_float32(Y, "Y");
    check_rank(A, 2, "A");
    check_rank(B, 2, "B");
    check_rank(Y, 2, "Y");

    const Shape &a_shape = A.shape();
    const Shape &b_shape = B.shape();
    const Shape &y_shape = Y.shape();
    if (a_shape[1] != b_shape[0] || y_shape[0] != a_shape[0] || y_shape[1] != b_shape[1])
        throw std::invalid_argument("Matmul: incompatible shapes");

    const size_t M = static_cast<size_t>(a_shape[0]);
    const size_t K = static_cast<size_t>(a_shape[1]);
    const size_t N = static_cast<size_t>(b_shape[1]);

    std::vector<float> out(M * N);
    matmul_optimized(A.data<float>(), B.data<float>(), out.data(), M, K, N);
    Y.set_data<float>(out);
}

void run_conv2d(const Tensor &X, const Tensor &W, Tensor &Y, std::int64_t stride_h,
                std::int64_t stride_w, std::int64_t pad_h, std::int64_t pad_w) {
    check_float32(X, "X");
    check_float32(W, "W");
    check_float32(Y, "Y");
    check_rank(X, 4, "X");
    check_rank(W, 4, "W");
    check_rank(Y, 4, "Y");

    const Shape &x_shape = X.shape();
    const Shape &w_shape = W.shape();
    const Shape &y_shape = Y.shape();
    if (x_shape[0] != 1 || y_shape[0] != 1)
        throw std::invalid_argument("Conv2d: only batch size 1 is supported");
    if (x_shape[1] != w_shape[1])
        throw std::invalid_argument("Conv2d: X and W channel dimensions are incompatible");
    if (y_shape[1] != w_shape[0])
        throw std::invalid_argument("Conv2d: Y channels must match W output channels");

    const std::int64_t C_in = x_shape[1];
    const std::int64_t H = x_shape[2];
    const std::int64_t Wid = x_shape[3];
    const std::int64_t C_out = w_shape[0];
    const std::int64_t KH = w_shape[2];
    const std::int64_t KW = w_shape[3];

    const std::int64_t OH = (H + 2 * pad_h - KH) / stride_h + 1;
    const std::int64_t OW = (Wid + 2 * pad_w - KW) / stride_w + 1;
    if (y_shape[2] != OH || y_shape[3] != OW)
        throw std::invalid_argument("Conv2d: Y shape does not match convolution parameters");

    std::vector<float> out(static_cast<size_t>(C_out * OH * OW));
    conv2d_im2col(X.data<float>(), W.data<float>(), out.data(), C_in, H, Wid, C_out, KH, KW,
                  stride_h, stride_w, pad_h, pad_w);
    Y.set_data<float>(out);
}

void run_matmul_node(const MatMulNode &node, Tensor &Y) {
    std::vector<TensorNode *> inputs = node.inputs();
    if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr)
        throw std::invalid_argument("Matmul: inputs are incomplete");

    const Tensor *A = inputs[0]->tensor();
    const Tensor *B = inputs[1]->tensor();
    if (A == nullptr || B == nullptr)
        throw std::invalid_argument("Matmul: tensors are not bound");
    run_matmul(*A, *B, Y);
}

void run_conv_node(const ConvNode &node, Tensor &Y) {
    std::vector<TensorNode *> inputs = node.inputs();
    if (inputs.size() < 2 || inputs[0] == nullptr || inputs[1] == nullptr)
        throw std::invalid_argument("Conv: inputs are incomplete");

    const Tensor *X = inputs[0]->tensor();
    const Tensor *W = inputs[1]->tensor();
    if (X == nullptr || W == nullptr)
        throw std::invalid_argument("Conv: tensors are not bound");

    const std::vector<std::int64_t> &strides = node.getStrides();
    const std::vector<std::int64_t> &pads = node.getPads();
    if (strides.size() != 2 || pads.size() != 4)
        throw std::invalid_argument("Conv: only 2D convolution is supported");
    run_conv2d(*X, *W, Y, strides[0], strides[1], pads[0], pads[1]);
}

} // namespace tensor_compiler::kernels
