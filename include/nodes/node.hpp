#pragma once

#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>
#include <string>
#include "viz/drawable.hpp"
#include "nodes/types.hpp"

namespace tensor_compiler {

class Node : public Drawable {

protected:
    std::string name_;
    NodeType type_;

public:
    Node(std::string name, NodeType type) : name_(std::move(name)), type_(type) {}
    virtual ~Node() override = default;

    const std::string &name() const {
        return name_;
    }
    NodeType type() const {
        return type_;
    }

    static std::string type2string(NodeType type) {
        switch (type) {
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

    static std::string type2shape(NodeType type) {
        if (type == NodeType::TENSOR)
            return "box";
        return "ellipse";
    }

    virtual Agnode_t *draw(Agraph_t *g) const override {
        std::string label = node_type_string(type_) + "\\n(" + name_ + ")";
        std::string shape = node_type_shape(type_);
        return Drawable::draw(g, label, shape);
    }
};
} // namespace tensor_compiler