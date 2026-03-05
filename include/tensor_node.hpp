#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "node.hpp"
#include "tensor.hpp"
#include <graphviz/cgraph.h>
#include <gvc.h>

namespace tensor_compiler {

template <typename NodePtr = Node *, typename TensorPtr = Tensor *> 
class TensorNode : public Node {
    TensorPtr tensor_{};

    NodePtr input_{};
    std::vector<NodePtr> output_{};

    void check_tensor() const {
        if (!tensor_)
            throw std::logic_error("Tensor is not initialized")
    }

public:
    TensorNode(std::string name, TensorPtr tensor = {}, NodePtr input = {},
               std::vector<NodePtr> output = {})
        : Node(name, Node::Type::TENSOR), tensor_(std::move(tensor)), input_(std::move(input)),
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

    const std::vector<NodePtr> &output() const {
        return output_;
    }

    void set_tensor(TensorPtr tensor) {
        if (tensor_)
            throw std::logic_error("Tensor is already tied");
        tensor_ = std::move(tensor);
    }

    void set_input(NodePtr node) {
        if (input_)
            throw std::logic_error("Input node is already tied");
        input_ = std::move(node);
    }

    void add_output(NodePtr output) {
        output_.push_back(std::move(output));
    }
};

} // namespace tensor_compiler
