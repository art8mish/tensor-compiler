#pragma once
#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>
#include <string>

namespace tensor_compiler {

enum class NodeType { CONV, GEMM, MATMUL, ADD, MUL, RELU, TENSOR };

std::string node_type_string(NodeType NodeType) {
        switch (NodeType) {
        case NodeType::CONV:
            return "Conv";
        case NodeType::GEMM:
            return "Gemm";
        case NodeType::MATMUL:
            return "MatMul";
        case NodeType::ADD:
            return "Add";
        case NodeType::MUL:
            return "Mul";
        case NodeType::RELU:
            return "ReLU";
        case NodeType::TENSOR:
            return "Tensor";
        default:
            return "Unknown";
        }
    }

std::string node_type_shape(NodeType NodeType) {
    if (NodeType == NodeType::TENSOR)
        return "box";
    return "ellipse";
}

} // namespace tensor_compiler