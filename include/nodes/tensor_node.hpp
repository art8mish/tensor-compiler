#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

#include "nodes/node.hpp"
#include "tensor/tensor.hpp"
#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

namespace tensor_compiler {

class TensorNode : public Node {
    Tensor *tensor_{};

    Node *input_{};
    std::unordered_set<Node *> output_{};

    void check_tensor() const {
        if (!tensor_)
            throw std::logic_error("Tensor is not initialized");
    }


public:
    TensorNode(std::string name, Tensor *tensor = {}, Node * input = {},
               std::unordered_set<Node *> output = {})
        : Node(name, NodeType::TENSOR), tensor_(tensor), input_(input),
          output_(std::move(output)) {}

    const Shape &shape() const {
        check_tensor();
        return tensor_->shape();
    }

    DataType dtype() const {
        check_tensor();
        return tensor_->dtype();
    }

    const Tensor *tensor() const {
        return tensor_;
    }

    const Node *input() const {
        return input_;
    }

    const std::unordered_set<Node *> &output() const {
        return output_;
    }

    void set_tensor(Tensor *tensor) {
        if (tensor_)
            throw std::logic_error("Tensor is already tied");
        tensor_ = tensor;
    }

    void set_input(Node *node) {
        if (input_)
            throw std::logic_error("Input node is already tied");
        input_ = node;
    }

    void add_output(Node *output) {
        output_.insert(output);
    }

    Agnode_t *draw(Agraph_t *g) const override {
        Agnode_t *node = (tensor_) ? tensor_->get_node(g) : get_node(g);
        if (node != nullptr)
            return node;
        node = (tensor_) ? tensor_->draw(g, name_) : Node::draw(g);
        for (const auto &out : output_) {
            Agnode_t *out_node = out->draw(g);
            agedge(g, node, out_node, nullptr, 1);
        }
        return node;
    }
};

} // namespace tensor_compiler
