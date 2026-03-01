
#include "compiler.hpp"
#include <iostream>

int main() {
    try {
        std::string expression;
        std::getline(std::cin, expression);
        if (!std::cin.good()) {
            std::cout << "Error: incorrect expression" << std::endl;
            return 1;
        }

#ifndef NDEBUG
        auto start_time = std::clock();
#endif
        // body
#ifndef NDEBUG
        auto duration = std::clock() - start_time;
        std::cout << "Runtime: " << duration << " us" << std::endl;
#endif
        return 0;
    } catch (const std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "Unknown error" << std::endl;
        return 2;
    }
}