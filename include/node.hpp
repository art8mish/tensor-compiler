#pragma once
#include <string>

#include <graphviz/cgraph.h>
#include <gvc.h>

namespace tensor_compiler {
    class Node {
        
    public:
        enum class Type { CONV, GEMM, MATMUL, ADD, MUL, RELU, TENSOR };
    protected:
        std::string name_;
        Type type_;

        std::string id() const {
            return std::to_string(reinterpret_cast<size_t>(this));
        }

        Agnode_t *draw_this(Agraph_t *g) const {
            std::string node_id = "node_" + id();
            Agnode_t *node = agnode(g, const_cast<char *>(node_id.c_str()), 1);

            std::string label = type2string(type_) + " (" + name_ + ")";
            agsafeset(node, const_cast<char *>("label"), const_cast<char *>(label.c_str()),
                    const_cast<char *>(""));
            std::string shape = type2shape(type_);
            agsafeset(node, const_cast<char *>("shape"), const_cast<char *>(shape.c_str()),
                    const_cast<char *>(""));
            return node;
        }

    public:
        Node(const std::string &name, Type otypep) : name_(n), type_(type) {}
        virtual ~Node() = default;
        virtual Agnode_t *draw(Agraph_t *g) const = 0;

        std::string type2string(Type type) {
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
                    return "?";
            }
        }

        std::string type2shape(Type type) {
            switch (type) {
                case Type::CONV:
                case Type::GEMM:
                case Type::MATMUL:
                case Type::ADD:
                case Type::MUL:
                case Type::RELU:
                    return "ellipse";
                case Type::TENSOR:
                    return "rectangle";
                default:
                    return "Error";string
            }
        }
    };
} // namespace tensor_compiler
