#pragma once

#include "node.hpp"
#include "tensor.hpp"
#include <list>
#include <memory>
#include <type_traits>

namespace tensor_compiler {

class ComputeGraph {
    std::list<std::unique_ptr<Node>> nodes_;
    std::list<std::unique_ptr<Tensor>> tensors_;

public:
    ComputeGraph() = default;
    ComputeGraph(const ComputeGraph &) = delete;
    ComputeGraph &operator=(const ComputeGraph &) = delete;

    template <typename NodeT, typename... Args> NodeT *add_node(Args &&...args) {
        static_assert(std::is_base_of<Node, NodeT>::value, "Type is not derived from Node");

        auto node = std::make_unique<NodeT>(std::forward<Args>(args)...);
        NodeT *ptr = node.get();
        nodes_.push_back(std::move(node));
        return ptr;
    }

    template <typename... Args> Tensor *add_tensor(Args &&...args) {
        auto tensor = std::make_unique<Tensor>(std::forward<Args>(args)...);
        Tensor *ptr = tensor.get();
        tensors_.push_back(std::move(tensor));
        return ptr;
    }

    void draw(Agraph_t *g) const {
        for (const auto &node : nodes_) {
            node->draw(g);
        }
    }

    size_t node_count() const {
        return nodes_.size();
    }
    size_t tensor_count() const {
        return tensors_.size();
    }
};

} // namespace tensor_compiler