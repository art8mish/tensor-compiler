#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "nodes/node.hpp"
#include "nodes/tensor_node.hpp"
#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

namespace tensor_compiler {

class OpNode : public Node {

protected:
    Shape out_shape_{};
    TensorNode *out_Y{};

    virtual Shape calc_out_shape() const = 0;

public:
    OpNode(std::string name, NodeType type) : Node(std::move(name), type) {}
    virtual ~OpNode() override = default;

    const Shape &get_out_shape() {
        if (out_shape_.empty())
            out_shape_ = calc_out_shape();
        return out_shape_;
    }

    void set_out_tensor(TensorNode *out_tensor) {
        if (out_Y)
            throw std::logic_error("Tensor is already tied");
        Shape out_shape = get_out_shape();
        if (out_tensor->shape() != out_shape)
            throw std::invalid_argument("Output tensor is incompatible with out shape");
        out_shape_ = out_shape;
        out_Y = out_tensor;
    }

    Agnode_t *draw(Agraph_t *g) const override {
        Agnode_t *node = get_node(g);
        if (node != nullptr)
            return node;

        node = Node::draw(g);
        if (out_Y) {
            Agnode_t *out_node = out_Y->draw(g);
            agedge(g, node, out_node, nullptr, 1);
        }
        return node;
    }

    virtual std::vector<TensorNode *> inputs() const = 0;
    const TensorNode *output() const {
        return out_Y;
    }
};

class ConvNode : public OpNode {
    TensorNode * in_X;
    TensorNode * in_W;
    TensorNode * in_Bias; // can be nullptr

    size_t dim_;
    std::vector<int64_t> kernel_shape_; // [x1_size, x2_size, ...]
    std::vector<int64_t> strides_;       // [x1_stride, x2_stride, ...]
    std::vector<int64_t> dilations_;     // [x1_dilation, x2_dilation, ...]
    std::vector<int64_t> pads_;         // [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    uint64_t group_ = 1;

    void validate() {
        if (!in_X || !in_W)
            throw std::invalid_argument("ConvNode: input nodes must be provided");

        if (dim_ < 1)
            throw std::invalid_argument("ConvNode: kernel dimension can't be a zero");

        if (strides_.size() != dim_)
            throw std::invalid_argument(
                "ConvNode: strides dimension is incompatible with kernel dimension");

        if (dilations_.size() != dim_)
            throw std::invalid_argument(
                "ConvNode: dilations dimension is incompatible with kernel dimension");

        if (pads_.size() != dim_ * 2)
            throw std::invalid_argument(
                "ConvNode: pads dimension is incompatible with kernel dimension");

        const auto &x_shape = in_X->shape();
        const auto &w_shape = in_W->shape();

        if (x_shape.size() != w_shape.size())
            throw std::invalid_argument("ConvNode: X and W shape sizes are incompatible");

        for (size_t i = 0; i < dim_; ++i)
            if (w_shape[i + 2] != static_cast<dim_t>(kernel_shape_[i]))
                throw std::invalid_argument(
                    "ConvNode: Weight shape is incompatible with kernel shape");

        // X_channels == W_channels * groups
        // x_shape[1] = in_Channels, w_shape[1] =  in_Channels / groups
        if (x_shape[1] != w_shape[1] * static_cast<dim_t>(group_))
            throw std::invalid_argument("ConvNode: Input channels are incompatible with groups");

        if (in_Bias) {
            const auto &b_shape = in_Bias->shape();
            // w_shape[0] = out_Channels
            if (b_shape.size() != 1 || b_shape[0] != w_shape[0])
                throw std::invalid_argument(
                    "ConvNode: Bias shape is incompatible with output channels");
        }
    }

    Shape calc_out_shape() const override {
        const auto &x_shape = in_X->shape();
        const auto &w_shape = in_W->shape();

        Shape out_shape;
        out_shape.reserve(2 + dim_);
        out_shape.push_back(x_shape[0]); // batch
        out_shape.push_back(w_shape[0]); // out_Channels

        for (size_t i = 0; i < dim_; ++i) {
            int64_t d_in = static_cast<int64_t>(x_shape[i + 2]);
            int64_t k = static_cast<int64_t>(w_shape[i + 2]);
            int64_t s = strides_[i];
            int64_t d = dilations_[i];
            int64_t pad_begin = pads_[i];
            int64_t pad_end = pads_[i + dim_];

            // kernel with delation
            int64_t del_k = d * (k - 1) + 1;

            if (d_in + pad_begin + pad_end < del_k)
                throw std::runtime_error("ConvNode: Input dimension with padding is too small for "
                                         "kernel with delations " +
                                         std::to_string(i));

            int64_t d_out = (d_in + pad_begin + pad_end - del_k) / s + 1;
            dim_t shape_out = static_cast<dim_t>(d_out);
            out_shape.push_back(shape_out);
        }

        return out_shape;
    }

public:
    ConvNode(std::string name, std::vector<int64_t> kernel_shape, TensorNode * X, TensorNode * W,
             TensorNode * bias = {}, std::vector<int64_t> strides = {},
             std::vector<int64_t> dilations = {}, std::vector<int64_t> pads = {},
             uint64_t group = 1)
        : OpNode(std::move(name), NodeType::CONV), in_X{X}, in_W{W}, in_Bias{bias},
          dim_{kernel_shape.size()}, kernel_shape_(std::move(kernel_shape)), group_(group) {
        if (strides.empty())
            strides_.assign(dim_, 1);
        else
            strides_ = std::move(strides);

        if (dilations.empty())
            dilations_.assign(dim_, 1);
        else
            dilations_ = std::move(dilations);

        if (pads.empty())
            pads_.assign(dim_ * 2, 0);
        else
            pads_ = std::move(pads);

        validate();
    }

    virtual std::vector<TensorNode *> inputs() const override {
        return {in_X, in_W, in_Bias};
    }

    const std::vector<int64_t>& getStrides() const { return strides_; }
    const std::vector<int64_t>& getDilations() const { return dilations_; }
    const std::vector<int64_t>& getPads() const { return pads_; }
};

// Gemm: Y = alpha*A*B + beta*C)
class GemmNode : public OpNode {
    TensorNode * in_A;
    TensorNode * in_B;
    TensorNode * in_C; // can be nullptr

    double alpha_;
    double beta_;
    bool trA_; // true -> transpose
    bool trB_;

    void validate() {
        if (!in_A || !in_B)
            throw std::invalid_argument("GemmNode: A and B are not provided");

        const auto &shape_A = in_A->shape();
        const auto &shape_B = in_B->shape();

        if (shape_A.size() != 2 || shape_B.size() != 2)
            throw std::invalid_argument("GemmNode: A and B are not a 2D tensors");

        dim_t A_rows = shape_A[0];
        dim_t A_cols = shape_A[1];
        dim_t B_rows = shape_B[0];
        dim_t B_cols = shape_B[1];

        dim_t i = trA_ ? shape_A[1] : shape_A[0]; // A rows
        dim_t k_A = trA_ ? A_rows : A_cols;       // A cols
        dim_t k_B = trB_ ? B_cols : B_rows;       // B rows
        dim_t j = trB_ ? shape_B[0] : shape_B[1]; // B cols

        if (k_A != k_B) {
            throw std::invalid_argument(
                "GemmNode: inner dimensions of A and B are incompatible after transpose");
        }

        if (in_C) {
            const auto &shape_C = in_C->shape();
            if (shape_C.size() != 2) {
                throw std::invalid_argument("GemmNode: C is not a 2D tensor");
            }
            if (shape_C[0] != i || shape_C[1] != j) {
                throw std::invalid_argument("GemmNode: C shape is incompatible with A and B");
            }
        }
    }

    Shape calc_out_shape() const override {
        Shape out_shape;
        out_shape.reserve(2);

        const auto &shape_A = in_A->shape();
        const auto &shape_B = in_B->shape();

        dim_t i = trA_ ? shape_A[1] : shape_A[0]; // A rows
        dim_t j = trB_ ? shape_B[0] : shape_B[1]; // B cols

        out_shape.push_back(i);
        out_shape.push_back(j);
        return out_shape;
    }

public:
    GemmNode(std::string name, TensorNode * A, TensorNode * B, TensorNode * C = {},
             double alpha = 1.0, double beta = 1.0, bool transpose_A = false,
             bool transpose_B = false)
        : OpNode(std::move(name), NodeType::GEMM), in_A{A}, in_B{B}, in_C{C}, alpha_{alpha},
          beta_{beta}, trA_{transpose_A}, trB_{transpose_B} {
        validate();
    }

    virtual std::vector<TensorNode *> inputs() const override {
        return {in_A, in_B, in_C};
    }

    // double get_alpha() const {
    //     return alpha_;
    // }
    // double get_beta() const {
    //     return beta_;
    // }
    // bool get_transpose_A() const {
    //     return trA_;
    // }
    // bool get_transpose_B() const {
    //     return trB_;
    // }
};

class MatMulNode : public OpNode {
    TensorNode * in_A;
    TensorNode * in_B;

    void validate() {
        if (!in_A || !in_B)
            throw std::invalid_argument("MatMulNode: input tensors are not provided");

        const auto &shape_A = in_A->shape();
        const auto &shape_B = in_B->shape();

        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();
        if (rank_A == 0 || rank_B == 0)
            throw std::invalid_argument("MatMulNode: tensors can't be scalars");

        dim_t k_A = shape_A[rank_A - 1];
        dim_t k_B = (rank_B == 1) ? shape_B[0] : shape_B[rank_B - 2];
        if (k_A != k_B)
            throw std::invalid_argument("MatMulNode: inner dimensions of inputs are incompatible");

        size_t batch_rank_A = (rank_A > 2) ? rank_A - 2 : 0;
        size_t batch_rank_B = (rank_B > 2) ? rank_B - 2 : 0;
        size_t min_batch_rank = std::min(batch_rank_A, batch_rank_B);

        if (min_batch_rank > 0) {
            for (size_t i = 1; i <= min_batch_rank; ++i) {
                dim_t dim_A = shape_A[batch_rank_A - i];
                dim_t dim_B = shape_B[batch_rank_B - i];
                if (dim_A != dim_B && dim_A != 1 && dim_B != 1)
                    throw std::invalid_argument(
                        "MatMulNode: batch dimensions are incompatible for broadcasting");
            }
        }
    }

    Shape calc_out_shape() const override {
        const Shape &shape_A = in_A->shape();
        const Shape &shape_B = in_B->shape();
        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();

        size_t batch_rank_A = (rank_A > 2) ? rank_A - 2 : 0;
        size_t batch_rank_B = (rank_B > 2) ? rank_B - 2 : 0;
        size_t batch_rank = std::max(batch_rank_A, batch_rank_B);

        Shape out_shape{};
        if (batch_rank > 0) {
            out_shape.resize(batch_rank, 1);

            for (size_t i = 1; i <= batch_rank; ++i) {
                dim_t dim_A = (i <= batch_rank_A) ? shape_A[batch_rank_A - i] : 1;
                dim_t dim_B = (i <= batch_rank_B) ? shape_B[batch_rank_B - i] : 1;
                out_shape[batch_rank - i] = std::max(dim_A, dim_B);
            }
        }

        if (rank_A >= 2)
            out_shape.push_back(shape_A[rank_A - 2]);
        if (rank_B >= 2)
            out_shape.push_back(shape_B[rank_B - 1]);

        return out_shape;
    }

public:
    MatMulNode(std::string name, TensorNode * A, TensorNode * B)
        : OpNode(std::move(name), NodeType::MATMUL), in_A{A}, in_B{B} {
        validate();
    }

    virtual std::vector<TensorNode *> inputs() const override {
        return {in_A, in_B};
    }
};

class AddNode : public OpNode {
    TensorNode * in_A;
    TensorNode * in_B;

    void validate() {
        if (!in_A || !in_B)
            throw std::invalid_argument("AddNode: input tensors are not provided");

        const auto &shape_A = in_A->shape();
        const auto &shape_B = in_B->shape();

        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();

        size_t min_rank = std::min(rank_A, rank_B);
        for (size_t i = 1; i <= min_rank; ++i) {
            dim_t dim_A = shape_A[rank_A - i];
            dim_t dim_B = shape_B[rank_B - i];

            if (dim_A != dim_B && dim_A != 1 && dim_B != 1) {
                throw std::invalid_argument(
                    "AddNode: dimensions at index " + std::to_string(i) +
                    " from end are incompatible for broadcasting: " + std::to_string(dim_A) +
                    " and " + std::to_string(dim_B));
            }
        }
    }

    Shape calc_out_shape() const override {
        const Shape &shape_A = in_A->shape();
        const Shape &shape_B = in_B->shape();
        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();
        size_t rank = std::max(rank_A, rank_B);

        Shape out_shape(rank);
        for (size_t i = 1; i <= rank; ++i) {
            dim_t dim_A = (i <= rank_A) ? shape_A[rank_A - i] : 1;
            dim_t dim_B = (i <= rank_B) ? shape_B[rank_B - i] : 1;

            if (dim_A != dim_B && dim_A != 1 && dim_B != 1)
                throw std::runtime_error("Incompatible shapes for Add operation");
            out_shape[rank - i] = std::max(dim_A, dim_B);
        }

        return out_shape;
    }

public:
    AddNode(std::string name, TensorNode * A, TensorNode * B)
        : OpNode(std::move(name), NodeType::ADD), in_A{A}, in_B{B} {
        validate();
    }

    virtual std::vector<TensorNode *> inputs() const override {
        return {in_A, in_B};
    }
};

class MulNode : public OpNode {
    TensorNode * in_A;
    TensorNode * in_B;

    void validate() {
        if (!in_A || !in_B)
            throw std::invalid_argument("AddNode: input tensors are not provided");

        const auto &shape_A = in_A->shape();
        const auto &shape_B = in_B->shape();

        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();

        size_t min_rank = std::min(rank_A, rank_B);
        for (size_t i = 1; i <= min_rank; ++i) {
            dim_t dim_A = shape_A[rank_A - i];
            dim_t dim_B = shape_B[rank_B - i];

            if (dim_A != dim_B && dim_A != 1 && dim_B != 1) {
                throw std::invalid_argument(
                    "MulNode: dimensions at index " + std::to_string(i) +
                    " from end are incompatible for broadcasting: " + std::to_string(dim_A) +
                    " and " + std::to_string(dim_B));
            }
        }
    }

    Shape calc_out_shape() const override {
        const Shape &shape_A = in_A->shape();
        const Shape &shape_B = in_B->shape();
        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();
        size_t rank = std::max(rank_A, rank_B);

        Shape out_shape(rank);
        for (size_t i = 1; i <= rank; ++i) {
            dim_t dim_A = (i <= rank_A) ? shape_A[rank_A - i] : 1;
            dim_t dim_B = (i <= rank_B) ? shape_B[rank_B - i] : 1;

            if (dim_A != dim_B && dim_A != 1 && dim_B != 1)
                throw std::runtime_error("Incompatible shapes for Mul operation");
            out_shape[rank - i] = std::max(dim_A, dim_B);
        }

        return out_shape;
    }

public:
    MulNode(const std::string &name, TensorNode * A, TensorNode * B)
        : OpNode(name, NodeType::MUL), in_A{A}, in_B{B} {
        validate();
    }

    virtual std::vector<TensorNode *> inputs() const override {
        return {in_A, in_B};
    }
};

class ReluNode : public OpNode {
    TensorNode * in_A;

    void validate() {
        if (!in_A)
            throw std::invalid_argument("ReluNode: input tensor is not provided");
    }

    Shape calc_out_shape() const override {
        return in_A->shape();
    }


public:
    ReluNode(const std::string &name, TensorNode * A) : OpNode(name, NodeType::RELU), in_A{A} {
        validate();
    }

    virtual std::vector<TensorNode *> inputs() const override {
        return {in_A};
    }
};

} // namespace tensor_compiler
