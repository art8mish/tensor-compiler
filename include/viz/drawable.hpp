#pragma once
#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>
#include <string>

namespace tensor_compiler {

class Drawable {
    virtual std::string id() const {
        return std::to_string(reinterpret_cast<uintptr_t>(this));
    }

    std::string node_id() const {
        return "n" + id();
    }

public:
    virtual ~Drawable() = default;

    virtual Agnode_t *draw(Agraph_t *g) const = 0;
    
    Agnode_t *draw(Agraph_t *g, const std::string &label, const std::string &shape) const {
        std::string n_id = node_id();
        Agnode_t *node = agnode(g, const_cast<char *>(n_id.c_str()), 1);

        char* label_str = nullptr;
        if (!label.empty() && label.front() == '<')
            label_str = agstrdup_html(g, const_cast<char *>(label.c_str()));
        else
            label_str = agstrdup(g, const_cast<char *>(label.c_str()));
        
        agsafeset(node, const_cast<char *>("label"), label_str, const_cast<char *>(""));
        agsafeset(node, const_cast<char *>("shape"), const_cast<char *>(shape.c_str()), const_cast<char *>(""));
        return node;
    }

    Agnode_t *get_node(Agraph_t *g) const {
        std::string n_id = node_id();
        return agnode(g, const_cast<char *>(n_id.c_str()), 0);
    }
};

} // namespace tensor_compiler