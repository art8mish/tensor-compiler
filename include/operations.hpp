#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "node.hpp"
#include "tensor_node.hpp"
#include <graphviz/cgraph.h>
#include <gvc.h>

namespace tensor_compiler {

template <typename TensorNodePtr = TensorNode *> class OpNode : public Node {
    Shape out_shape_{};
    TensorNodePtr out_Y{};

    virtual Shape calc_out_shape() const = 0;

public:
    OpNode(std::string name, Type type) : Node(std::move(name), type) {}

    const Shape &get_out_shape() const {
        if (out_shape_.empty())
            out_shape_ = calc_out_shape();
        return out_shape_;
    }

    void set_out_tensor(TensTensorNodePtr out_tensor) {
        if (out_Y)
            throw std::logic_error("Tensor is already tied");
        out_shape = get_out_shape();
        if (out_tensor->shape() != out_shape)
            throw std::invalid_argument("Output tensor is incompatible with out shape");
        out_Y = out_tensor;
    }

    Agnode_t *draw(Agraph_t *g) const override {
        Agnode_t *node = draw_this();
        if (out_Y) {
            Agnode_t *out_node = out_Y->draw(g);
            agedge(g, node, out_node, nullptr, 1);
        }
        return node;
    }
};

template <typename TensorNodePtr = TensorNode *> class ConvNode : public OpNode {
    TensTensorNodePtr in_X;
    TensTensorNodePtr in_W;
    TensTensorNodePtr in_Bias; // can be nullptr

    size_t dim_;
    std::vector<uint64_t> kernel_shape_; // [x1_size, x2_size, ...]
    std::vector<uint64_t> strides_;      // [x1_stride, x2_stride, ...]
    std::vector<uint64_t> dilations_;    // [x1_dilation, x2_dilation, ...]
    std::vector<uint64_t> pads_;         // [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
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

        const auto &x_shape = in_X->shape;
        const auto &w_shape = in_W->shape;

        if (x_shape.size() != w_shape.size())
            throw std::invalid_argument("ConvNode: X and W shape sizes are incompatible");

        for (size_t i = 0; i < dim_; ++i)
            if (w_shape[i + 2] != kernel_shape_[i])
                throw std::invalid_argument(
                    "ConvNode: Weight shape is incompatible with kernel shape");

        // X_channels == W_channels * groups
        // x_shape[1] = in_Channels, w_shape[1] =  in_Channels / groups
        if (x_shape[1] != w_shape[1] * group)
            throw std::invalid_argument("ConvNode: Input channels are incompatible with groups");

        if (in_Bias) {
            const auto &b_shape = in_Bias->shape;
            // w_shape[0] = out_Channels
            if (b_shape.size() != 1 || b_shape[0] != w_shape[0])
                throw std::invalid_argument(
                    "ConvNode: Bias shape is incompatible with output channels");
        }
    }

    Shape calc_out_shape() {
        const auto &x_shape = in_X->shape;
        const auto &w_shape = in_W->shape;

        Shape out_shape;
        out_shape.reserve(2 + dim_);
        out_shape.push_back(x_shape[0]); // batch
        out_shape.push_back(w_shape[0]); // out_Channels

        for (size_t i = 0; i < dim_; ++i) {
            uint64_t d_in = x_shape[i + 2];
            uint64_t k = w_shape[i + 2];
            uint64_t s = strides_[i];
            uint64_t d = dilations_[i];
            uint64_t pad_begin = pads[i];
            uint64_t pad_end = pads[i + dim_];

            // kernel with delation
            uint64_t del_k = d * (k - 1) + 1;

            if (d_in + pad_begin + pad_end < del_k)
                throw std::runtime_error("ConvNode: Input dimension with padding is too small for "
                                         "kernel with delations " +
                                         std::to_string(i));

            uint64_t d_out = (d_in + pad_begin + pad_end - del_k) / s + 1;
            dim_t shape_out = static_cast<dim_t>(d_out);
            out_shape.push_back(shape_out);
        }

        return out_shape;
    }

public:
    ConvNode(std::string name, std::vector<uint64_t> kernel_shape, TensTensorNodePtr X, TensTensorNodePtr W,
             TensTensorNodePtr bias = {}, std::vector<uint64_t> strides = {},
             std::vector<uint64_t> dilations = {}, std::vector<uint64_t> pads = {},
             uint64_t group = 1)
        : OpNode(std::move(name), Node::Type::CONV), in_X{std::move(X)}, in_W{std::move(W)},
          in_Bias{std::move(bias)}, dim_{kernel_shape.size()},
          kernel_shape_(std::move(kernel_shape)), group_(g) {
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
};

// Gemm: Y = alpha*A*B + beta*C)
template <typename TensorNodePtr = TensorNode *> class GemmNode : public OpNode {
    TensTensorNodePtr in_A;
    TensTensorNodePtr in_B;
    TensTensorNodePtr in_C; // can be nullptr

    double alpha_;
    double beta_;
    bool trA_; // true -> transpose
    bool trB_;

    void validate() {
        if (!in_A || !in_B)
            throw std::invalid_argument("GemmNode: A and B are not provided");

        const auto &shape_A = in_A->shape;
        const auto &shape_B = in_B->shape;

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
            const auto &shape_C = in_C->shape;
            if (shape_C.size() != 2) {
                throw std::invalid_argument("GemmNode: C is not a 2D tensor");
            }
            if (shape_C[0] != i || shape_C[1] != j) {
                throw std::invalid_argument("GemmNode: C shape is incompatible with A and B");
            }
        }
    }

    Shape calc_out_shape() {
        Shape out_shape;
        out_shape.reserve(2);

        const auto &shape_A = in_A->shape;
        const auto &shape_B = in_B->shape;

        dim_t i = trA_ ? shape_A[1] : shape_A[0]; // A rows
        dim_t j = trB_ ? shape_B[0] : shape_B[1]; // B cols

        out_shape.push_back(i);
        out_shape.push_back(j);
        return out_shape;
    }

public:
    GemmNode(std::string name, TensTensorNodePtr A, TensTensorNodePtr B, TensTensorNodePtr C = {}, double alpha = 1.0,
             double beta = 1.0, bool transpose_A = false, bool transpose_B = false)
        : OpNode(std::move(name), Node::Type::GEMM), in_A{std::move(A)}, in_B{std::move(B)},
          in_C{std::move(C)}, alpha_{alpha}, beta_{beta}, trA_{transpose_A}, trB_{transpose_B} {
        validate();
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

template <typename TensorNodePtr = TensorNode *> class MatMulNode : public OpNode {
    TensTensorNodePtr in_A;
    TensTensorNodePtr in_B;

    void validate() {
        if (!in_A || !in_B)
            throw std::invalid_argument("MatMulNode: input tensors are not provided");

        const auto &shape_A = in_A->shape;
        const auto &shape_B = in_B->shape;

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
            size_t i_A = batch_rank_A - 1;
            size_t i_B = batch_rank_B - 1;
            for (size_t i = min_batch_rank - 1; i >= 0; --i) {
                dim_t dim_A = shape_A[i_A--];
                dim_t dim_B = shape_B[i_B--];
                if (dim_A != dim_B && dim_A != 1 && dim_B != 1)
                    throw std::invalid_argument(
                        "MatMulNode: batch dimensions are incompatible for broadcasting");
            }
        }
    }

    Shape calc_out_shape() {
        const Shape &shape_A = in_A->shape;
        const Shape &shape_B = in_B->shape;
        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();

        size_t batch_rank_A = (rank_A > 2) ? rank_A - 2 : 0;
        size_t batch_rank_B = (rank_B > 2) ? rank_B - 2 : 0;
        size_t batch_rank = std::max(batch_rank_A, batch_rank_B)

            Shape out_shape{};
        if (batch_rank > 0) {
            out_shape.resize(batch_rank, 1);

            size_t i_A = (batch_rank_A > 1) ? batch_rank_A - 1 : 0;
            size_t i_B = (batch_rank_B > 1) ? batch_rank_B - 1 : 0;
            for (size_t i = batch_rank - 1; i >= 0; --i) {
                dim_t dim_A = (i_A >= 0) ? shape_A[i_A--] : 1;
                dim_t dim_B = (i_B >= 0) ? shape_B[i_B--] : 1;
                out_shape[i] = std::max(dim_A, dim_B);
            }
        }

        if (rank_A >= 2)
            out_shape.push_back(shape_A[rank_A - 2]);
        if (rank_B >= 2)
            out_shape.push_back(shape_B[rank_B - 1]);

        return out_shape;
    }

public:
    MatMulNode(std::string name, TensTensorNodePtr A, TensTensorNodePtr B)
        : OpNode(std::move(name), Node::Type::MATMUL), in_A{std::move(A)}, in_B{std::move(B)} {
        validate();
    }
};

template <typename TensorNodePtr = TensorNode *> class AddNode : public OpNode {
    TensTensorNodePtr in_A;
    TensTensorNodePtr in_B;

    void validate() {
        if (!in_A || !in_B)
            throw std::invalid_argument("AddNode: input tensors are not provided");

        const auto &shape_A = in_A->shape;
        const auto &shape_B = in_B->shape;

        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();
        size_t min_rank = std::min(rank_A, rank_B);

        if (min_rank > 0) {
            size_t i_A = rank_A - 1;
            size_t i_B = rank_B - 1;
            for (size_t i = min_rank - 1; i >= 0; --i) {
                dim_t dim_A = shape_A[i_A--];
                dim_t dim_B = shape_B[i_B--];
                if (dim_A != dim_B && dim_A != 1 && dim_B != 1)
                    throw std::invalid_argument(
                        "AddNode: dimensions are incompatible for broadcasting");
            }
        }
    }

    Shape calc_out_shape() {
        const Shape &shape_A = in_A->shape;
        const Shape &shape_B = in_B->shape;
        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();
        size_t rank = std::max(rank_A, rank_B)

            Shape out_shape{};
        if (rank > 0) {
            out_shape.resize(rank, 1);

            size_t i_A = (rank_A > 1) ? rank_A - 1 : 0;
            size_t i_B = (rank_B > 1) ? rank_B - 1 : 0;
            for (size_t i = batch_rank - 1; i >= 0; --i) {
                dim_t dim_A = (i_A >= 0) ? shape_A[i_A--] : 1;
                dim_t dim_B = (i_B >= 0) ? shape_B[i_B--] : 1;
                out_shape[i] = std::max(dim_A, dim_B);
            }
        }

        return out_shape;
    }

public:
    AddNode(std::string name, TensTensorNodePtr A, TensTensorNodePtr B)
        : OpNode(std::move(name), Node::Type::ADD), in_A{std::move(A)}, in_B{std::move(B)} {
        validate();
    }
};

template <typename TensorNodePtr = TensorNode *> class MulNode : public OpNode {
    TensTensorNodePtr in_A;
    TensTensorNodePtr in_B;

    void validate() {
        if (!in_A || !in_B)
            throw std::invalid_argument("MulNode: input tensors are not provided");

        const auto &shape_A = in_A->shape;
        const auto &shape_B = in_B->shape;

        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();
        size_t min_rank = std::min(rank_A, rank_B);

        if (min_rank > 0) {
            size_t i_A = rank_A - 1;
            size_t i_B = rank_B - 1;
            for (size_t i = min_rank - 1; i >= 0; --i) {
                dim_t dim_A = shape_A[i_A--];
                dim_t dim_B = shape_B[i_B--];
                if (dim_A != dim_B && dim_A != 1 && dim_B != 1)
                    throw std::invalid_argument(
                        "MulNode: dimensions are incompatible for broadcasting");
            }
        }
    }

    Shape calc_out_shape() {
        const Shape &shape_A = in_A->shape;
        const Shape &shape_B = in_B->shape;
        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();
        size_t rank = std::max(rank_A, rank_B)

            Shape out_shape{};
        if (rank > 0) {
            out_shape.resize(rank, 1);

            size_t i_A = (rank_A > 1) ? rank_A - 1 : 0;
            size_t i_B = (rank_B > 1) ? rank_B - 1 : 0;
            for (size_t i = batch_rank - 1; i >= 0; --i) {
                dim_t dim_A = (i_A >= 0) ? shape_A[i_A--] : 1;
                dim_t dim_B = (i_B >= 0) ? shape_B[i_B--] : 1;
                out_shape[i] = std::max(dim_A, dim_B);
            }
        }

        return out_shape;
    }

public:
    MulNode(const std::string &name, TensTensorNodePtr A, TensTensorNodePtr B)
        : OpNode(name, Node::Type::MUL), in_A{A}, in_B{B} {
        validate();
    }
};

template <typename TensorNodePtr = TensorNode *> class ReluNode : public OpNode {
    TensTensorNodePtr in_A;

    void validate() {
        if (!in_A)
            throw std::invalid_argument("ReluNode: input tensor is not provided");
    }

    Shape calc_out_shape() {
        const Shape &shape_A = in_A->shape;
        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();
        size_t rank = std::max(rank_A, rank_B)

            Shape out_shape{};
        if (rank > 0) {
            out_shape.resize(rank, 1);

            size_t i_A = (rank_A > 1) ? rank_A - 1 : 0;
            size_t i_B = (rank_B > 1) ? rank_B - 1 : 0;
            for (size_t i = batch_rank - 1; i >= 0; --i) {
                dim_t dim_A = (i_A >= 0) ? shape_A[i_A--] : 1;
                dim_t dim_B = (i_B >= 0) ? shape_B[i_B--] : 1;
                out_shape[i] = std::max(dim_A, dim_B);
            }
        }

        return out_shape;
    }

public:
    ReluNode(const std::string &name, TensTensorNodePtr A) : OpNode(name, Node::Type::RELU), in_A{A} {
        validate();
    }
};

} // namespace tensor_compiler
