#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

#include "node.hpp"
#include "node.hpp"
#include "tensor.hpp"
#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

namespace tensor_compiler {

template <typename NodePtr = Node *, typename TensorPtr = Tensor *> class TensorNode : public Node {
    TensorPtr tensor_{};

    NodePtr input_{};
    std::unordered_set<NodePtr> output_{};

    void check_tensor() const {
        if (!tensor_)
            throw std::logic_error("Tensor is not initialized");
    }

public:
    TensorNode(std::string name, TensorPtr tensor = {}, NodePtr input = {},
               std::unordered_set<NodePtr> output = {})
        : Node(name, Node::Type::TENSOR), tensor_(tensor), input_(input),
          output_(std::move(output)) {}

    const Shape &shape() const {
        check_tensor();
        return tensor_->shape();
    }

    DataType dtype() const {
        check_tensor();
        return tensor_->dtype();
    }

    const TensorPtr &tensor() const {
        return tensor_;
    }

    const NodePtr &input() const {
        return input_;
    }

    const std::unordered_set<NodePtr> &output() const {
        return output_;
    }

    void set_tensor(TensorPtr tensor) {
        if (tensor_)
            throw std::logic_error("Tensor is already tied");
        tensor_ = tensor;
    }

    void set_input(NodePtr node) {
        if (input_)
            throw std::logic_error("Input node is already tied");
        input_ = node;
    }

    void add_output(NodePtr output) {
        output_.insert(output);
    }

    Agnode_t *draw(Agraph_t *g) const override {
        Agnode_t *node = get_node(g);
        if (node != nullptr)
            return node;

        node = draw_this(g);
        for (const auto &out: output_) {
            Agnode_t *out_node = out->draw(g);
            agedge(g, node, out_node, nullptr, 1);
        }
        return node;
    }
};

} // namespace tensor_compiler
