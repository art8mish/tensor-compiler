#pragma once

#include "nodes/node.hpp"
#include "nodes/tensor_node.hpp"
#include "nodes/operations.hpp"
#include "tensor/tensor.hpp"
#include <list>
#include <memory>
#include <type_traits>
#include <ranges>

namespace tensor_compiler {

class ComputeGraph {
    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<std::unique_ptr<Tensor>> tensors_;

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


    auto nodes() const {
        return nodes_ | std::views::transform([](const auto& ptr) -> const Node* {
            return ptr.get();
        });
    }

    auto tensors() const {
        return tensors_ | std::views::transform([](const auto& ptr) -> const Tensor* {
            return ptr.get();
        });
    }

    size_t node_count() const {
        return nodes_.size();
    }
    size_t tensor_count() const {
        return tensors_.size();
    }

    void draw(Agraph_t *g) const {
        for (const auto &node : nodes_)
            node->draw(g);
    }
};

} // namespace tensor_compiler