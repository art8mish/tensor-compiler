#pragma once

#include <unordered_map>
#include <vector>
#include "mlir_compute_graph.hpp"
#include "nodes/operations.hpp"
#include "nodes/tensor_node.hpp"

// MLIR includes
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace tensor_compiler {

class MLIRBuilder {
public:
    mlir::OpBuilder builder_;
    std::unordered_map<const Node*, mlir::Value> value_map_;
    MLIRBuilder(mlir::MLIRContext* ctx) : builder_(ctx) {}

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

    mlir::RankedTensorType get_tensor_type(const TNodePtr tensor) {
        std::vector<int64_t> shape;
        for (auto d : tensor->shape()) 
            shape.push_back(d == DYNAMIC_DIM ? mlir::ShapedType::kDynamic : d);
        return mlir::RankedTensorType::get(shape, convert_type(tensor->dtype()));
    }

    template <typename T>
    std::vector<T> cast_data(const TNodePtr tensor) {
        const auto& tensor = tensor->tensor();
        std::vector<T> promoted;
        promoted.reserve(tensor->size());

        auto convert = [&](auto* ptr) {
            for (size_t i = 0; i < tensor->size(); ++i) 
                promoted.push_back(static_cast<T>(ptr[i]));
        };

        switch (tensor->dtype()) {
            case DataType::FLOAT32: convert(tensor->data<DataType_t<DataType::FLOAT32>>()); break;
            case DataType::FLOAT64: convert(tensor->data<DataType_t<DataType::FLOAT64>>()); break;
            case DataType::INT64:   convert(tensor->data<DataType_t<DataType::INT64>>()); break;
            case DataType::INT32:   convert(tensor->data<DataType_t<DataType::INT32>>()); break;
            case DataType::INT8:    convert(tensor->data<DataType_t<DataType::INT8>>()); break;
            case DataType::UINT8:   convert(tensor->data<DataType_t<DataType::UINT8>>()); break;
            case DataType::BOOL:    convert(tensor->data<DataType_t<DataType::BOOL>>()); break;
            default: break;
        }
        return promoted;
    }

    mlir::Value tensor_value(const TNodePtr tensor) {
        mlir::RankedTensorType type = get_tensor_type(tensor);
        std::vector<float> data = cast_data<float>(tensor);
        mlir::DenseElementsAttr attr = mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(data));

        // TODO: different types
        // if (type.getElementType().isF32()) {
        //     auto data = cast_data<float>(tensor);
        //     attr = mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(data));
        // } else {
        //     ...
        // }
        return builder_.create<mlir::arith::ConstantOp>(builder_.getUnknownLoc(), attr);
    }

    

    void process_node(const Node* node) {
        mlir::Location loc = builder_.getUnknownLoc();

        if (node->type() == NodeType::TENSOR) {
            const TNodePtr t_node = static_cast<const TNodePtr>(node);
            if (t_node->tensor() && !t_node->tensor()->empty() && !value_map_.contains(node))
                value_map_[node] = tensor_value(t_node); 
            return;
        }

        const OpNodePtr op_node = static_cast<const OpNodePtr>(node);
        const TensorNodePtr out_tensor = op_node->output();
        mlir::RankedTensorType out_tensor_type = get_tensor_type(out_tensor);
        
        mlir::Value out_format = builder_.create<mlir::tensor::EmptyOp>(
            loc, out_tensor_type.getShape(), out_tensor_type.getElementType());

        auto node_type = node->type();
        auto inputs = op_node->inputs();
        mlir::Value result;

        switch (node_type) {
            case NodeType::RELU: {
                mlir::Value in_tensor = value_map_.at(inputs[0]);
                result = builder_.create<mlir::linalg::MapOp>(loc, mlir::ValueRange{in_tensor}, out_format,
                    [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
                        auto zero = b.create<mlir::arith::ConstantOp>(l, b.getFloatAttr(args[0].getType(), 0.0));
                        auto max = b.create<mlir::arith::MaximumFOp>(l, args[0], zero);
                        b.create<mlir::linalg::YieldOp>(l, max);
                    }).getResult(0);
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
                    result = builder_.create<mlir::linalg::AddOp>(loc, 
                        mlir::ValueRange{in_A, in_B}, mlir::ValueRange{out_format}).getResult(0);
                else if (node_type == NodeType::MUL)
                    result = builder_.create<mlir::linalg::MulOp>(loc, 
                        mlir::ValueRange{in_A, in_B}, mlir::ValueRange{out_format}).getResult(0);
                else
                    result = builder_.create<mlir::linalg::MatmulOp>(loc, 
                        mlir::ValueRange{in_A, in_B}, mlir::ValueRange{out_format}).getResult(0);
                break;
            }

            case NodeType::CONV: {
                auto* convNode = static_cast<const ConvNode<TensorNodePtr>*>(node);
                mlir::Value X = value_map_.at(inputs[0]);
                mlir::Value W = value_map_.at(inputs[1]);
                
                auto strides = builder_.getI64ArrayAttr(convNode->getStrides());
                auto dilations = builder_.getI64ArrayAttr(convNode->getDilations());

                result = builder_.create<mlir::linalg::Conv2DNhwcFhwcOp>(
                    loc, mlir::ValueRange{X, W}, mlir::ValueRange{out_format}, strides, dilations
                ).getResult(0);

                if (inputs.size() > 2 && inputs[2]) {
                    mlir::Value bias = value_map_.at(inputs[2]);
                    result = builder_.create<mlir::linalg::AddOp>(loc, 
                        mlir::ValueRange{result, bias}, mlir::ValueRange{result}).getResult(0);
                }
                break;
            }

            default:
                throw std::runtime_error("Unsupported NodeType");
        }

        if (result) {
            value_map_[out_tensor] = result;
        }
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

    //     llvm::SmallVector<mlir::utils::IteratorType, 4> iter_types(out_rank, mlir::utils::IteratorType::parallel);

    //     return b.create<mlir::linalg::GenericOp>(
    //         loc, 
    //         out_type, 
    //         mlir::ValueRange{lhs, rhs},
    //         mlir::ValueRange{out_init},
    //         b.getAffineMapArrayAttr({lhs_map, rhs_map, out_map}),
    //         b.getArrayAttr(llvm::to_vector<8>(llvm::map_range(iter_types, [&](mlir::utils::IteratorType t) {
    //             return mlir::utils::IteratorTypeAttr::get(b.getContext(), t);
    //         }))),
    //         [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
    //             mlir::Value result_op = b.create<OpT>(l, args[0], args[1]);
    //             b.create<mlir::linalg::YieldOp>(l, result_op);
    //         }
    //     ).getResult(0);
    // }

    // void process_node(const Node* node) {
    //     mlir::Location loc = builder_.getUnknownLoc();

    //     if (node->type() == NodeType::TENSOR) {
    //         const TNodePtr t_node = static_cast<const TNodePtr>(node);
    //         const TensorPtr tensor = t_node->tensor();
    //         if (tensor && !tensor->empty() && !value_map_.contains(node))
    //             value_map_[node] = tensor_value(t_node); 
    //         return;
    //     }

    //     const OpNodePtr op_node = static_cast<const OpNodePtr>(node);
    //     const TensorNodePtr out_tensor = op_node->output();
    //     mlir::RankedTensorType out_tensor_type = get_tensor_type(out_tensor);
        
    //     mlir::Value out_format = builder_.create<mlir::tensor::EmptyOp>(
    //         loc, out_tensor_type.getShape(), out_tensor_type.getElementType());

    //     auto node_type = node->type();
    //     auto inputs = op_node->inputs();
    //     mlir::Value result;

    //     switch (node_type) {
    //         case NodeType::RELU: {
    //             mlir::Value in_tensor = value_map_.at(inputs[0]);
    //             result = builder_.create<mlir::linalg::MapOp>(loc, mlir::ValueRange{in_tensor}, out_format,
    //                 [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
    //                     auto zero = b.create<mlir::arith::ConstantOp>(l, b.getFloatAttr(args[0].getType(), 0.0));
    //                     auto max = b.create<mlir::arith::MaximumFOp>(l, args[0], zero);
    //                     b.create<mlir::linalg::YieldOp>(l, max);
    //                 }).getResult(0);
    //             break;
    //         }

    //         case NodeType::ADD:
    //         case NodeType::MUL: {
    //             mlir::Value lhs = value_map_.at(inputs[0]);
    //             mlir::Value rhs = value_map_.at(inputs[1]);
                
    //             if (node_type == NodeType::ADD)
    //                 result = broadcast<mlir::arith::AddFOp>(builder_, loc, lhs, rhs, out_format);
    //             else
    //                 result = broadcast<mlir::arith::MulFOp>(builder_, loc, lhs, rhs, out_format);
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
    //                 mlir::ValueRange{in_A, in_B}, mlir::ValueRange{out_format}).getResult(0);
    //             break;
    //         }

    //         case NodeType::CONV: {
    //             auto convNode = static_cast<const ConvNode<TensorNodePtr>*>(node);
    //             mlir::Value X = value_map_.at(inputs[0]);
    //             mlir::Value W = value_map_.at(inputs[1]);
                
    //             auto strides = builder_.getI64ArrayAttr(convNode->getStrides());
    //             auto dilations = builder_.getI64ArrayAttr(convNode->getDilations());

    //             mlir::Value result = builder_.create<mlir::linalg::Conv2DNhwcFhwcOp>(
    //                 loc, mlir::ValueRange{X, W}, mlir::ValueRange{out_format}, strides, dilations
    //             ).getResult(0);

    //             if (inputs.size() > 2 && inputs[2]) {
    //                 mlir::Value bias = value_map_.at(inputs[2]);
    //                 result = broadcast<mlir::arith::AddFOp>(builder_, loc, result, bias, conv_res);
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