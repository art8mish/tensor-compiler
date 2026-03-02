#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "node.hpp"
#include <graphviz/cgraph.h>
#include <gvc.h>

namespace tensor_compiler {

    using ShapeType = uint64_t;
    using Shape = std::vector<ShapeType>;

    template <typename NodeIt>
    class TensorNode : public Node {
        std::vector<uint64_t> shape_;

    TensorNode(std::string name, std::initializer_list<int64_t> dims) : 
        : Node(name, NodeType::TENSOR), shape_(dims) {}
};
   
} // namespace tensor_compiler
