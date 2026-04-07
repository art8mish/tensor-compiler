#pragma once
#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>
#include <string>

namespace tensor_compiler {
class Node {
public:
    enum class Type { CONV, GEMM, MATMUL, ADD, MUL, RELU, TENSOR };

protected:
    std::string name_;
    Type type_;

    std::string id() const {
        return std::to_string(reinterpret_cast<uintptr_t>(this));
    }

    std::string node_id() const {
        return "n" + id();
    }

    Agnode_t *draw_this(Agraph_t *g) const {
        std::string n_id = node_id();
        Agnode_t *node = agnode(g, const_cast<char *>(n_id.c_str()), 1);

        std::string label_str = type2string(type_) + "\\n(" + name_ + ")";
        agsafeset(node, const_cast<char *>("label"), const_cast<char *>(label_str.c_str()), const_cast<char *>(""));
        agsafeset(node, const_cast<char *>("shape"), const_cast<char *>(type2shape(type_).c_str()), const_cast<char *>(""));

        return node;
    }

    Agnode_t *get_node(Agraph_t *g) const {
        std::string n_id = node_id();
        return agnode(g, const_cast<char *>(n_id.c_str()), 0);
    }

public:
    Node(std::string name, Type type) : name_(std::move(name)), type_(type) {}
    virtual ~Node() = default;

    virtual Agnode_t *draw(Agraph_t *g) const = 0;

    const std::string &name() const {
        return name_;
    }
    Type type() const {
        return type_;
    }

    static std::string type2string(Type type) {
        switch (type) {
        case Type::CONV:
            return "Conv";
        case Type::GEMM:
            return "Gemm";
        case Type::MATMUL:
            return "MatMul";
        case Type::ADD:
            return "Add";
        case Type::MUL:
            return "Mul";
        case Type::RELU:
            return "ReLU";
        case Type::TENSOR:
            return "Tensor";
        default:
            return "Unknown";
        }
    }

    static std::string type2shape(Type type) {
        if (type == Type::TENSOR)
            return "box";
        return "ellipse";
    }
};
} // namespace tensor_compiler