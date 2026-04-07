#pragma once
#include <iostream>
#include <stdexcept>
#include <string>

#include "cgraph.hpp"
#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

namespace tensor_compiler {

void draw_graph(const ComputeGraph &graph, const std::string &output_file) {
    GVC_t *gvc = gvContext();
    if (!gvc)
        throw std::runtime_error("Failed to create Graphviz context");

    Agraph_t *g = agopen(const_cast<char *>("G"), Agdirected, nullptr);
    if (!g) {
        gvFreeContext(gvc);
        throw std::runtime_error("Failed to open graph");
    }

    agsafeset(g, const_cast<char *>("rankdir"), const_cast<char *>("TB"), const_cast<char *>(""));
    agsafeset(g, const_cast<char *>("nodesep"), const_cast<char *>("0.5"), const_cast<char *>(""));
    agsafeset(g, const_cast<char *>("overlap"), const_cast<char *>("false"), const_cast<char *>(""));
    agsafeset(g, const_cast<char *>("splines"), const_cast<char *>("true"), const_cast<char *>(""));

    try {
        graph.draw(g);

        if (gvLayout(gvc, g, "dot") != 0)
            throw std::runtime_error("Graphviz layout failed");

        std::string format = "png";
        if (output_file.find(".svg") != std::string::npos) format = "svg";
        else if (output_file.find(".pdf") != std::string::npos) format = "pdf";
        else if (output_file.find(".dot") != std::string::npos) format = "canon";

        if (gvRenderFilename(gvc, g, format.c_str(), output_file.c_str()) != 0)
            throw std::runtime_error("Graphviz rendering failed");

    } catch (const std::exception &e) {
        gvFreeLayout(gvc, g);
        agclose(g);
        gvFreeContext(gvc);
        throw;
    }

    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);
}

} // namespace tensor_compiler