#pragma once
#include <iostream>

#include "node.hpp"
#include <graphviz/cgraph.h>
#include <gvc.h>

namespace tensor_compiler {

void draw_graph(const Node *root, const std::string &output_file) {
    if (!root)
        return;

    GVC_t *gvc = gvContext();
    Agraph_t *g = agopen(const_cast<char *>("AST"), Agdirected, nullptr);

    agsafeset(g, const_cast<char *>("rankdir"), const_cast<char *>("TB"), const_cast<char *>(""));

    root->draw(g);

    gvLayout(gvc, g, "dot");
    gvRenderFilename(gvc, g, "png", output_file.c_str());

    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);
}

} // namespace tensor_compiler
