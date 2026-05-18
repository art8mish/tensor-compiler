#pragma once

#include "mlir_ext/graph.hpp"
#include "nodes/node.hpp"
#include "nodes/operations.hpp"
#include "nodes/tensor_node.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

namespace tensor_compiler {

class MLIRBuilder {
public:
    mlir::OpBuilder builder_;
    std::unordered_map<const Node *, mlir::Value> value_map_;
    MLIRBuilder(mlir::MLIRContext *ctx) : builder_(ctx) {}

private:
public:
    mlir::Type convert_type(DataType dtype) {
        return builder_.getF32Type();
        // switch (dtype) {
        //     case DataType::FLOAT32:
        //     case DataType::FLOAT64:
        //     case DataType::BOOL:
        //         return builder_.getF32Type();

        //     case DataType::INT32:
        //     case DataType::INT64:
        //     case DataType::INT8:
        //     case DataType::UINT8:
        //         return builder_.getI32Type();

        //     default:
        //         return builder_.getF32Type();
        // }
    }

    mlir::RankedTensorType get_tensor_type(const TensorNode *tensor) {
        std::vector<std::int64_t> shape;
        for (auto d : tensor->shape())
            shape.push_back(d == DYNAMIC_DIM ? mlir::ShapedType::kDynamic : d);
        return mlir::RankedTensorType::get(shape, convert_type(tensor->dtype()));
    }

    template <typename T> std::vector<T> cast_data(const TensorNode *tensor_node) {
        const auto &tensor = tensor_node->tensor();
        std::vector<T> casted;
        casted.reserve(tensor->size());

        auto convert = [&](auto *ptr) {
            for (size_t i = 0; i < tensor->size(); ++i)
                casted.push_back(static_cast<T>(ptr[i]));
        };

        switch (tensor->dtype()) {
        case DataType::FLOAT32:
            convert(tensor->template data<DataType_t<DataType::FLOAT32>>());
            break;
        case DataType::FLOAT64:
            convert(tensor->template data<DataType_t<DataType::FLOAT64>>());
            break;
        case DataType::INT64:
            convert(tensor->template data<DataType_t<DataType::INT64>>());
            break;
        case DataType::INT32:
            convert(tensor->template data<DataType_t<DataType::INT32>>());
            break;
        case DataType::INT8:
            convert(tensor->template data<DataType_t<DataType::INT8>>());
            break;
        case DataType::UINT8:
            convert(tensor->template data<DataType_t<DataType::UINT8>>());
            break;
        case DataType::BOOL:
            convert(tensor->template data<DataType_t<DataType::BOOL>>());
            break;
        default:
            break;
        }
        return casted;
    }

    mlir::Value tensor_value(const TensorNode *tensor) {
        mlir::RankedTensorType type = get_tensor_type(tensor);
        std::vector<float> data = cast_data<float>(tensor);
        mlir::DenseElementsAttr attr =
            mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(data));

        // TODO: different types
        // if (type.getElementType().isF32()) {
        //     auto data = cast_data<float>(tensor);
        //     attr = mlir::DenseElementsAttr::get(type,
        //     llvm::ArrayRef<float>(data));
        // } else {
        //     ...
        // }
        return mlir::arith::ConstantOp::create(builder_, builder_.getUnknownLoc(), attr);
    }

    mlir::Value gen_matmul_generic(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                                   mlir::Value output_acc, mlir::RankedTensorType out_ty) {
        auto lhs_rt = mlir::cast<mlir::RankedTensorType>(lhs.getType());
        const unsigned rank = static_cast<unsigned>(lhs_rt.getRank());
        const unsigned batch_rank = rank - 2;
        const unsigned n_iter = batch_rank + 3;

        mlir::MLIRContext *ctx = builder_.getContext();
        llvm::SmallVector<mlir::AffineExpr> lhs_exprs, rhs_exprs, out_exprs;

        for (unsigned i = 0; i < batch_rank; ++i) {
            mlir::AffineExpr di = builder_.getAffineDimExpr(i);
            lhs_exprs.push_back(di);
            rhs_exprs.push_back(di);
            out_exprs.push_back(di);
        }
        lhs_exprs.push_back(builder_.getAffineDimExpr(batch_rank));
        lhs_exprs.push_back(builder_.getAffineDimExpr(n_iter - 1));

        rhs_exprs.push_back(builder_.getAffineDimExpr(n_iter - 1));
        rhs_exprs.push_back(builder_.getAffineDimExpr(batch_rank + 1));

        out_exprs.push_back(builder_.getAffineDimExpr(batch_rank));
        out_exprs.push_back(builder_.getAffineDimExpr(batch_rank + 1));

        mlir::AffineMap lhs_map = mlir::AffineMap::get(n_iter, 0, lhs_exprs, ctx);
        mlir::AffineMap rhs_map = mlir::AffineMap::get(n_iter, 0, rhs_exprs, ctx);
        mlir::AffineMap out_map = mlir::AffineMap::get(n_iter, 0, out_exprs, ctx);

        llvm::SmallVector<mlir::AffineMap> maps = {lhs_map, rhs_map, out_map};
        llvm::SmallVector<mlir::utils::IteratorType, 8> iterators(
            n_iter, mlir::utils::IteratorType::parallel);
        iterators.back() = mlir::utils::IteratorType::reduction;

        return mlir::linalg::GenericOp::create(
                   builder_, loc, mlir::TypeRange{out_ty}, mlir::ValueRange{lhs, rhs},
                   mlir::ValueRange{output_acc}, maps, iterators,
                   [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
                       auto prod = mlir::arith::MulFOp::create(b, l, args[0], args[1]);
                       auto acc = mlir::arith::AddFOp::create(b, l, args[2], prod.getResult());
                       mlir::linalg::YieldOp::create(b, l, acc.getResult());
                   })
            .getResult(0);
    }

    /// ONNX pads [begin_dim0, begin_dim1, …, end_dim0, end_dim1, …]
    mlir::Value emit_onnx_pads(mlir::Location loc, mlir::Value X,
                               const std::vector<std::int64_t> &pads, unsigned spatial_rank) {
        if (pads.size() != static_cast<size_t>(spatial_rank) * 2)
            throw std::runtime_error("Conv pads list length must be 2 * spatial_rank");

        if (llvm::all_of(pads, [](std::int64_t p) { return p == 0; }))
            return X;

        auto rt = mlir::cast<mlir::RankedTensorType>(X.getType());
        const std::int64_t rank = rt.getRank();
        if (rank < 2 || static_cast<size_t>(rank) < 2 + spatial_rank)
            throw std::runtime_error("Conv input rank incompatible with pads");

        llvm::SmallVector<std::int64_t> new_shape(rt.getShape().begin(), rt.getShape().end());
        llvm::SmallVector<std::int64_t> low_pad(static_cast<size_t>(rank), 0);
        for (unsigned i = 0; i < spatial_rank; ++i) {
            const unsigned dim_idx = static_cast<unsigned>(rank) - spatial_rank + i;
            const std::int64_t pad_before = pads[i];
            const std::int64_t pad_after = pads[i + spatial_rank];
            new_shape[dim_idx] += pad_before + pad_after;
            low_pad[dim_idx] = pad_before;
        }

        mlir::RankedTensorType padded_ty =
            mlir::RankedTensorType::get(new_shape, rt.getElementType());

        mlir::Value scratch = mlir::tensor::EmptyOp::create(builder_, loc, padded_ty.getShape(),
                                                            padded_ty.getElementType())
                                  .getResult();
        mlir::FloatType elem_ty = mlir::cast<mlir::FloatType>(padded_ty.getElementType());
        mlir::Value zero =
            mlir::arith::ConstantFloatOp::create(builder_, loc, elem_ty, llvm::APFloat(0.0f));
        mlir::Value filled =
            mlir::linalg::FillOp::create(builder_, loc, mlir::TypeRange{padded_ty},
                                         mlir::ValueRange{zero}, mlir::ValueRange{scratch})
                .getResult(0);

        llvm::SmallVector<std::int64_t> src_sizes(rt.getShape().begin(), rt.getShape().end());
        llvm::SmallVector<std::int64_t> strides(static_cast<size_t>(rank), 1);

        return mlir::tensor::InsertSliceOp::create(
                   builder_, loc, padded_ty, X, filled, mlir::ValueRange{}, mlir::ValueRange{},
                   mlir::ValueRange{}, llvm::ArrayRef<std::int64_t>(low_pad),
                   llvm::ArrayRef<std::int64_t>(src_sizes), llvm::ArrayRef<std::int64_t>(strides))
            .getResult();
    }

    void finalize(const Node *output_node) {
        mlir::Location loc = builder_.getUnknownLoc();

        auto func = llvm::dyn_cast<mlir::func::FuncOp>(builder_.getInsertionBlock()->getParentOp());

        mlir::ValueRange final_value{};

        if (value_map_.contains(output_node))
            final_value = mlir::ValueRange{value_map_.at(output_node)};

        if (func) {
            auto new_type = builder_.getFunctionType(func.getArgumentTypes(), final_value);
            func.setType(new_type);
        }

        mlir::func::ReturnOp::create(builder_, loc, final_value);
    }

    void process_node(const Node *node) {
        mlir::Location loc = builder_.getUnknownLoc();

        if (node->type() == NodeType::TENSOR) {
            const TensorNode *t_node = dynamic_cast<const TensorNode *>(node);
            if (t_node->tensor() && !t_node->tensor()->empty() && !value_map_.contains(node))
                value_map_[node] = tensor_value(t_node);
            return;
        }

        const OpNode *op_node = static_cast<const OpNode *>(node);
        const TensorNode *out_tensor = op_node->output();
        mlir::RankedTensorType out_tensor_type = get_tensor_type(out_tensor);

        mlir::Value out_format = mlir::tensor::EmptyOp::create(
            builder_, loc, out_tensor_type.getShape(), out_tensor_type.getElementType());

        auto node_type = node->type();
        auto inputs = op_node->inputs();
        mlir::Value result;

        switch (node_type) {
        case NodeType::RELU: {
            mlir::Value in_tensor = value_map_.at(inputs[0]);
            const auto in_rt = mlir::cast<mlir::RankedTensorType>(in_tensor.getType());
            const unsigned rank = static_cast<unsigned>(in_rt.getRank());
            mlir::MLIRContext *ctx = builder_.getContext();
            mlir::AffineMap id = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
            llvm::SmallVector<mlir::AffineMap> maps(2, id);
            llvm::SmallVector<mlir::utils::IteratorType, 4> iterators(
                rank, mlir::utils::IteratorType::parallel);
            result = mlir::linalg::GenericOp::create(
                         builder_, loc, mlir::TypeRange{out_tensor_type},
                         mlir::ValueRange{in_tensor}, mlir::ValueRange{out_format}, maps, iterators,
                         [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
                             auto zero = mlir::arith::ConstantOp::create(
                                 b, l, b.getFloatAttr(args[0].getType(), 0.0));
                             auto max = mlir::arith::MaximumFOp::create(b, l, args[0], zero);
                             mlir::linalg::YieldOp::create(b, l, max.getResult());
                         })
                         .getResult(0);
            break;
        }

        case NodeType::GEMM:
            if (inputs.size() > 2 && inputs[2])
                out_format = value_map_.at(inputs[2]);
            [[fallthrough]];
        case NodeType::ADD:
            [[fallthrough]];
        case NodeType::MUL:
            [[fallthrough]];
        case NodeType::MATMUL: {
            mlir::Value in_A = value_map_.at(inputs[0]);
            mlir::Value in_B = value_map_.at(inputs[1]);
            if (node_type == NodeType::ADD)
                result = mlir::linalg::AddOp::create(builder_, loc, {out_format.getType()},
                                                     mlir::ValueRange{in_A, in_B},
                                                     mlir::ValueRange{out_format})
                             .getResult(0);
            else if (node_type == NodeType::MUL)
                result = mlir::linalg::MulOp::create(builder_, loc, {out_format.getType()},
                                                     mlir::ValueRange{in_A, in_B},
                                                     mlir::ValueRange{out_format})
                             .getResult(0);
            else {
                const unsigned lhs_rank = static_cast<unsigned>(
                    mlir::cast<mlir::RankedTensorType>(in_A.getType()).getRank());

                const bool gemm_use_c_as_init =
                    (node_type == NodeType::GEMM && inputs.size() > 2 && inputs[2]);

                mlir::Value matmul_out_acc = out_format;
                if (!gemm_use_c_as_init) {
                    mlir::Value zero = mlir::arith::ConstantFloatOp::create(
                        builder_, loc, builder_.getF32Type(), llvm::APFloat(0.0f));
                    matmul_out_acc = mlir::linalg::FillOp::create(
                                         builder_, loc, mlir::TypeRange{out_tensor_type},
                                         mlir::ValueRange{zero}, mlir::ValueRange{out_format})
                                         .getResult(0);
                }

                if (lhs_rank <= 2)
                    result = mlir::linalg::MatmulOp::create(
                                 builder_, loc, mlir::TypeRange{out_tensor_type},
                                 mlir::ValueRange{in_A, in_B}, mlir::ValueRange{matmul_out_acc})
                                 .getResult(0);
                else
                    result = gen_matmul_generic(loc, in_A, in_B, matmul_out_acc, out_tensor_type);
            }
            break;
        }

        case NodeType::CONV: {
            auto *convNode = static_cast<const ConvNode *>(node);
            mlir::Value X = value_map_.at(inputs[0]);
            mlir::Value W = value_map_.at(inputs[1]);

            const auto &onnx_pads = convNode->getPads();
            const unsigned spatial_rank = static_cast<unsigned>(onnx_pads.size() / 2);
            X = emit_onnx_pads(loc, X, onnx_pads, spatial_rank);

            auto strides = builder_.getI64ArrayAttr(convNode->getStrides());
            auto dilations = builder_.getI64ArrayAttr(convNode->getDilations());

            mlir::Value conv_out_init = out_format;
            if (inputs.size() > 2 && inputs[2])
                conv_out_init =
                    conv_bias(loc, out_tensor_type, static_cast<const TensorNode *>(inputs[2]));

            result = mlir::linalg::Conv2DNchwFchwOp::create(
                         builder_, loc, {out_tensor_type}, mlir::ValueRange{X, W},
                         mlir::ValueRange{conv_out_init}, strides, dilations, {})
                         .getResult(0);

            break;
        }

        default:
            throw std::runtime_error("Unsupported NodeType");
        }

        if (result) {
            value_map_[out_tensor] = result;
        }
    }

    mlir::Value conv_bias(mlir::Location loc, mlir::RankedTensorType conv_out_ty,
                          const TensorNode *bias_node) {
        const Tensor *bias_tensor = bias_node->tensor();
        if (!bias_tensor || bias_tensor->empty())
            throw std::runtime_error("Conv bias: tensor is missing or empty");
        if (bias_tensor->is_dynamic())
            throw std::runtime_error("Conv bias: dynamic shape is not supported");

        if (conv_out_ty.getRank() != 4)
            throw std::runtime_error("Conv bias expansion: expected 4D conv output type");
        for (std::int64_t d : conv_out_ty.getShape()) {
            if (d == mlir::ShapedType::kDynamic)
                throw std::runtime_error("Conv bias: dynamic conv output dimension");
        }

        std::vector<float> bias_data = cast_data<float>(bias_node);
        const std::int64_t c_out = conv_out_ty.getDimSize(1);
        if (static_cast<std::int64_t>(bias_data.size()) != c_out)
            throw std::runtime_error("Conv bias: expected " + std::to_string(c_out) +
                                     " values, got " + std::to_string(bias_data.size()));

        const size_t N = static_cast<size_t>(conv_out_ty.getDimSize(0));
        const size_t C = static_cast<size_t>(conv_out_ty.getDimSize(1));
        const size_t H = static_cast<size_t>(conv_out_ty.getDimSize(2));
        const size_t W = static_cast<size_t>(conv_out_ty.getDimSize(3));

        std::vector<float> expanded_bias(static_cast<size_t>(N * C * H * W));
        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < C; ++c) {
                const float bv = bias_data[c];
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        const size_t idx = ((n * C + c) * H + h) * W + w;
                        expanded_bias[idx] = bv;
                    }
                }
            }
        }

        mlir::DenseElementsAttr attr =
            mlir::DenseElementsAttr::get(conv_out_ty, llvm::ArrayRef<float>(expanded_bias));
        return mlir::arith::ConstantOp::create(builder_, loc, attr).getResult();
    }

    // TODO: broadcast

    // template <typename OpT>
    // mlir::Value broadcast(mlir::OpBuilder &b, mlir::Location loc,
    //         mlir::Value lhs, mlir::Value rhs, mlir::Value out_init) {
    //     auto out_type = out_init.getType().cast<mlir::RankedTensorType>();
    //     int64_t out_rank = outType.getRank();

    //     auto get_map = [&](mlir::Value val) -> mlir::AffineMap {
    //         auto type = val.getType().cast<mlir::RankedTensorType>();
    //         int64_t rank = type.getRank();
    //         auto shape = type.getShape();
    //         llvm::SmallVector<mlir::AffineExpr, 4> exprs;

    //         for (int64_t i = 0; i < rank; ++i) {
    //             int64_t out_i = out_rank - rank + i;

    //             if (shape[i] == 1)
    //                 exprs.push_back(b.getAffineConstantExpr(0));
    //             else
    //                 exprs.push_back(b.getAffineDimExpr(out_i));
    //         }
    //         return mlir::AffineMap::get(out_rank, 0, exprs, b.getContext());
    //     };

    //     mlir::AffineMap lhs_map = get_map(lhs);
    //     mlir::AffineMap rhs_map = get_map(rhs);
    //     mlir::AffineMap out_map = b.getMultiDimIdentityMap(out_rank);

    //     llvm::SmallVector<mlir::utils::IteratorType, 4> iter_types(out_rank,
    //     mlir::utils::IteratorType::parallel);

    //     return b.create<mlir::linalg::GenericOp>(
    //         loc,
    //         out_type,
    //         mlir::ValueRange{lhs, rhs},
    //         mlir::ValueRange{out_init},
    //         b.getAffineMapArrayAttr({lhs_map, rhs_map, out_map}),
    //         b.getArrayAttr(llvm::to_vector<8>(llvm::map_range(iter_types,
    //         [&](mlir::utils::IteratorType t) {
    //             return mlir::utils::IteratorTypeAttr::get(b.getContext(), t);
    //         }))),
    //         [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args)
    //         {
    //             mlir::Value result_op = b.create<OpT>(l, args[0], args[1]);
    //             b.create<mlir::linalg::YieldOp>(l, result_op);
    //         }
    //     ).getResult()[0];
    // }

    // void process_node(const Node* node) {
    //     mlir::Location loc = builder_.getUnknownLoc();

    //     if (node->type() == NodeType::TENSOR) {
    //         const TensorNode * t_node = static_cast<const TensorNode
    //         *>(node); const TensorPtr tensor = t_node->tensor(); if (tensor
    //         && !tensor->empty() && !value_map_.contains(node))
    //             value_map_[node] = tensor_value(t_node);
    //         return;
    //     }

    //     const OpNode * op_node = static_cast<const OpNode *>(node);
    //     const TensorNodePtr out_tensor = op_node->output();
    //     mlir::RankedTensorType out_tensor_type = get_tensor_type(out_tensor);

    //     mlir::Value out_format = builder_.create<mlir::tensor::EmptyOp>(
    //         loc, out_tensor_type.getShape(),
    //         out_tensor_type.getElementType());

    //     auto node_type = node->type();
    //     auto inputs = op_node->inputs();
    //     mlir::Value result;

    //     switch (node_type) {
    //         case NodeType::RELU: {
    //             mlir::Value in_tensor = value_map_.at(inputs[0]);
    //             result = builder_.create<mlir::linalg::MapOp>(loc,
    //             mlir::ValueRange{in_tensor}, out_format,
    //                 [&](mlir::OpBuilder &b, mlir::Location l,
    //                 mlir::ValueRange args) {
    //                     auto zero = b.create<mlir::arith::ConstantOp>(l,
    //                     b.getFloatAttr(args[0].getType(), 0.0)); auto max =
    //                     b.create<mlir::arith::MaximumFOp>(l, args[0], zero);
    //                     b.create<mlir::linalg::YieldOp>(l, max);
    //                 }).getResult()[0];
    //             break;
    //         }

    //         case NodeType::ADD:
    //         case NodeType::MUL: {
    //             mlir::Value lhs = value_map_.at(inputs[0]);
    //             mlir::Value rhs = value_map_.at(inputs[1]);

    //             if (node_type == NodeType::ADD)
    //                 result = broadcast<mlir::arith::AddFOp>(builder_, loc,
    //                 lhs, rhs, out_format);
    //             else
    //                 result = broadcast<mlir::arith::MulFOp>(builder_, loc,
    //                 lhs, rhs, out_format);
    //             break;
    //         }

    //         case NodeType::GEMM:
    //             if (inputs.size() > 2 && inputs[2]) {
    //                 out_format = value_map_.at(inputs[2]);
    //             }
    //             [[fallthrough]];
    //         case NodeType::MATMUL: {
    //             mlir::Value in_A = value_map_.at(inputs[0]);
    //             mlir::Value in_B = value_map_.at(inputs[1]);
    //             result = builder_.create<mlir::linalg::MatmulOp>(loc,
    //                 mlir::ValueRange{in_A, in_B},
    //                 mlir::ValueRange{out_format}).getResult()[0];
    //             break;
    //         }

    //         case NodeType::CONV: {
    //             auto convNode = static_cast<const
    //             ConvNode<TensorNodePtr>*>(node); mlir::Value X =
    //             value_map_.at(inputs[0]); mlir::Value W =
    //             value_map_.at(inputs[1]);

    //             auto strides =
    //             builder_.getI64ArrayAttr(convNode->getStrides()); auto
    //             dilations =
    //             builder_.getI64ArrayAttr(convNode->getDilations());

    //             mlir::Value result =
    //             builder_.create<mlir::linalg::Conv2DNhwcFhwcOp>(
    //                 loc, mlir::ValueRange{X, W},
    //                 mlir::ValueRange{out_format}, strides, dilations
    //             ).getResult()[0];

    //             if (inputs.size() > 2 && inputs[2]) {
    //                 mlir::Value bias = value_map_.at(inputs[2]);
    //                 result = broadcast<mlir::arith::AddFOp>(builder_, loc,
    //                 result, bias, conv_res);
    //             break;
    //         }

    //         default:
    //             throw std::runtime_error("Unsupported NodeType");
    //     }

    //     if (result) {
    //         value_map_[out_tensor] = result;
    //     }
    // }
};
} // namespace tensor_compiler