#pragma once

#include <vector>
#include "node.hpp"
#include "tensor.hpp"


namespace tensor_compiler {


class ComputeGraph {
    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<std::unique_ptr<Tensor>> tensors_;

public:
    ComputeGraph() = default;

    template<typename T, typename... Args>
    T* add_node(Args&&... args) {
        static_assert(std::is_base_of<Node, T>::value, "Type is not derived from Node");
        auto node = std::make_unique<T>(std::forward<Args>(args)...);
        T* node_ptr = node.get();
        nodes_.push_back(std::move(node));
        return node_ptr;
    }

    template<typename... Args>
    Tensor* add_tensor(Args&&... args) {
        auto tensor = std::make_unique<Tensor>(std::forward<Args>(args)...);
        Tensor* tensor_ptr = tensor.get();
        tensors_.push_back(std::move(tensor));
        return tensor_ptr;
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

};
} // namespace tensor_compiler