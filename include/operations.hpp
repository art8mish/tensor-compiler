#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "node.hpp"
#include <graphviz/cgraph.h>
#include <gvc.h>

namespace tensor_compiler {

template <typename NodePtr> class OpNode : public Node {
    Shape out_shape_{};
    NodePtr out_Y{};

    virtual Shape calc_out_shape() const = 0;

public:
    OpNode(const std::string &name, Type type) : Node(name, type) {}

    const Shape &get_out_shape() const {
        if (out_shape_.empty())
            out_shape_ = calc_out_shape();
        return out_shape_;
    }

    void set_out_tensor(NodePtr out_tensor) {
        out_shape = get_out_shape();
        if (out_tensor->shape() != out_shape)
            throw std::invalid_argument("Output tensor is incompatible with out shape");
        out_Y = out_tensor;
    }
};

template <typename NodePtr>
class ConvNode : public OpNode {
    NodePtr in_X;
    NodePtr in_W;
    NodePtr in_Bias; // can be nullptr

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
            ShapeType shape_out = static_cast<ShapeType>(d_out);
            out_shape.push_back(shape_out);
        }

        return out_shape;
    }

public:
    ConvNode(const std::string &name, const std::vector<uint64_t> &kernel_shape, NodePtr X, NodePtr W,
             NodePtr bias = {}, const std::vector<uint64_t> &strides = {},
             const std::vector<uint64_t> &dilations = {}, const std::vector<uint64_t> &pads = {},
             uint64_t group = 1)
        : OpNode(name, Node::Type::CONV), in_X{X}, in_W{W}, in_Bias{bias}, dim_{kernel_shape.size()},
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
template <typename NodePtr> class GemmNode : public Node {
    NodePtr in_A;
    NodePtr in_B;
    NodePtr in_C; // can be nullptr

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

        ShapeType A_rows = shape_A[0];
        ShapeType A_cols = shape_A[1];
        ShapeType B_rows = shape_B[0];
        ShapeType B_cols = shape_B[1];

        ShapeType i = trA_ ? shape_A[1] : shape_A[0]; // A rows
        ShapeType k_A = trA_ ? A_rows : A_cols;       // A cols
        ShapeType k_B = trB_ ? B_cols : B_rows;       // B rows
        ShapeType j = trB_ ? shape_B[0] : shape_B[1]; // B cols

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

        ShapeType i = trA_ ? shape_A[1] : shape_A[0]; // A rows
        ShapeType j = trB_ ? shape_B[0] : shape_B[1]; // B cols

        out_shape.push_back(i);
        out_shape.push_back(j);
        return out_shape;
    }

public:
    GemmNode(const std::string &name, NodePtr A, NodePtr B, NodePtr C = {}, double alpha = 1.0,
             double beta = 1.0, bool transpose_A = false, bool transpose_B = false)
        : OpNode(name, Node::Type::GEMM), in_A{A}, in_B{B}, in_C{C}, alpha_{alpha}, beta_{beta},
          trA_{transpose_A}, trB_{transpose_B} {
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

template <typename NodePtr> class MatMulNode : public Node {
    NodePtr in_A;
    NodePtr in_B;

    void validate() {
        if (!in_A || !in_B)
            throw std::invalid_argument("MatMulNode: input tensors are not provided");

        const auto &shape_A = in_A->shape;
        const auto &shape_B = in_B->shape;

        size_t rank_A = shape_A.size();
        size_t rank_B = shape_B.size();

        ShapeType k_A, k_B;
        ShapeType i = 1, j = 1;

        // both 1D -> scalar
        if (rank_A == 1 && rank_B == 1) {
            // Оба векторы: длина должна совпадать
            if (shape_A[0] != shape_B[0])
                throw std::invalid_argument(
                    "MatMulNode: 1D tensors shapes are incompatible");
            k_A = shape_A[0];
            k_B = shape_B[0];
        }

        //1=1D and 2>=2D -> scalar
        else if (rank_A == 1 && rank_B >= 2) {
            k_A = shape_A[0];
            k_B = shape_B[rank_B - 2]; // prelast B
            j = shape_B[rank_B - 1];
        }
        else if (rank_A >= 2 && rank_B == 1) {
            k_A = shape_A[rank_A - 1]; // last A
            k_B = shape_B[0];
            i = shape_A[rank_A - 2];   // prelast A
        }
        else if (rank_A >= 2 && rank_B >= 2) {
            k_A = shape_A[rank_A - 1]; // last A
            k_B = shape_B[rank_B - 2]; // prelast B
            i = shape_A[rank_A - 2];
            j = shape_B[rank_B - 1];
        }
        else
            throw std::invalid_argument("MatMulNode: invalid tensor ranks");

        if (k_A != k_B) {
            throw std::invalid_argument(
                "MatMulNode: inner dimensions of inputs are incompatible");
        }
    }

    Shape calc_out_shape() {
        Shape out_shape {};
        std::vector<ShapeType> batch_A, batch_B;
        if (rank_A == 1)
            batch_A = {};
        else
            for (size_t i = 0; i < rank_A - 2; ++i)
                batch_A.push_back(shape_A[i]);

        if (rank_B == 1)
            batch_B = {};
        else
            for (size_t i = 0; i < rank_B - 2; ++i)
                batch_B.push_back(shape_B[i]);

        size_t max_batch = std::max(batch_A.size(), batch_B.size());
        out_shape.resize(max_batch, 1);

        for (size_t i = 0; i < max_batch; ++i) {
            ShapeType dim_A = (i < batch_A.size()) ? batch_A[batch_A.size() - 1 - i] : 1;
            ShapeType dim_B = (i < batch_B.size()) ? batch_B[batch_B.size() - 1 - i] : 1;

            if (dim_A != dim_B && dim_A != 1 && dim_B != 1) {
                throw std::invalid_argument(
                    "MatMulNode: batch dimensions are incompatible for broadcasting");
            }

            // Результирующая размерность по этой оси — максимум
            out_shape[max_batch - 1 - i] = std::max(dim_A, dim_B);
        }


        ShapeType i = (rank_A >= 2) ? shape_A[rank_A - 2] : 1;
        ShapeType j = (rank_B >= 2) ? shape_B[rank_B - 1] : 1;
        if (i != 1 || j != 1) {
            out_shape.push_back(i);
            out_shape.push_back(j);
        }
        // Если out_shape пуст, это означает скаляр (0D тензор)
        return out_shape;
    }

public:
    MatMulNode(const std::string &name) : OpNode(name, Node::Type::MATMUL) {}
};

template <typename NodePtr> class AddNode : public OpNode {
    NodePtr in_A;
    NodePtr in_B;

public:
    AddNode(const std::string &name) : OpNode(name, Node::Type::ADD) {}
};

template <typename NodePtr> class MulNode : public OpNode {
    NodePtr in_A;
    NodePtr in_B;

public:
    MulNode(const std::string &name) : OpNode(name, Node::Type::MUL) {}
};

template <typename NodePtr> class ReluNode : public OpNode {
    NodePtr in_A;

public:
    ReluNode(const std::string &name) : OpNode(name, Node::Type::RELU) {}

    Agnode_t *draw(Agraph_t *g) const override {
        Agnode_t *node = draw_this();

        Agnode_t *out_node = out_Y->draw(g);
        agedge(g, node, out_node, nullptr, 1);

        return node;
    }
};

} // namespace tensor_compiler
