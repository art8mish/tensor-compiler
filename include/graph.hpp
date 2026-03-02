

#include <vector>
#include "node.hpp"


namespace tensor_compiler {


class ComputeGraph {
    std::vector<std::unique_ptr<Node>> nodes_;
    
public:
    ComputeGraph() = default;

    template<typename T, typename... Args>
    T* addNode(Args&&... args) {
        auto node = std::make_unique<T>(std::forward<Args>(args)...);
        T* ptr = node.get();
        nodes_.push_back(std::move(node));
        return ptr;
    }
};

} // namespace tensor_compiler