
#include "factory.hpp"
#include "viz/cg_draw.hpp"
#include <iostream>
#include <ctime>

int main() {
    try {
        // std::getline(std::cin, expression);
        // if (!std::cin.good()) {
        //     std::cout << "Error: incorrect expression" << std::endl;
        //     return 1;
        // }

        std::string filename = "end2end/models/test_model.onnx";
        tensor_compiler::ComputeGraphFactory factory;

#ifndef NDEBUG
        auto start_time = std::clock();
#endif
        auto graph = factory.from_onnx(filename);
#ifndef NDEBUG
        auto duration = std::clock() - start_time;
        std::cout << "Runtime: " << duration << " us" << std::endl;
#endif
        std::string graph_file = "results/graph.png";
        tensor_compiler::draw_compute_graph(*graph, graph_file);

        return 0;
    } catch (const std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "Unknown error" << std::endl;
        return 2;
    }
}