

#include "cli.hpp"
#include "cgraph.hpp"
#include "factory.hpp"
#include "mlir_ext/graph.hpp"
#include "mlir_ext/factory.hpp"
#include "mlir_ext/compiler.hpp"
#include "viz/cg_draw.hpp"
#include <ctime>
#include <filesystem>
#include <iostream>
#include <llvm/IR/LLVMContext.h>
#include <string>

namespace tc=tensor_compiler;

#ifndef NDEBUG
#endif

int main(int argc, char **argv) {
    try {
        tc::Cli parsed(argc, argv);
        if (parsed.help) {
            tc::Cli::print_help(std::cout, argv[0]);
            return 0;
        }

        auto start_time = std::clock();
        auto graph = tc::ComputeGraphFactory::from_onnx(*parsed.input_onnx);

        if (parsed.out_graph) {
            std::filesystem::path gp(*parsed.out_graph);
            if (!gp.has_parent_path())
                std::filesystem::create_directories(gp.parent_path());
            tc::draw_compute_graph(*graph, *parsed.out_graph);
            std::cout << "Graph written: " << *parsed.out_graph << std::endl;
        }

        auto mlir_g = tc::MLIRComputeGraphFactory::create(*graph, "mlir_graph");
    
        tc::MLIRCompiler compiler;
        auto llvm_ctx = compiler.compile(*mlir_g, *parsed.out_path, tc::OutputFormat::Object, parsed.llvm_opts, assembler*parsed.out_asm)
        if (!llvm_ctx) {
            std::cout << "Compilation failed\n";
            return 1;
        }
        
        llvm_mod->dump();
        if (!emit_file(*llvm_mod, out_path, format, llvm_opts))
            return false;
        if (assembly_path && !assembly_path->empty()) {
            if (!emit_file(*llvm_mod, *assembly_path, OutputFormat::Assembly, llvm_opts))
                return false;
        }

        std::cout << "Created: " << *parsed.out_path << std::endl;
        if (parsed.out_asm)
            std::cout << "Created: " << *parsed.out_asm << std::endl;

        auto duration = std::clock() - start_time;
        std::cout << "Runtime: " << duration << " us" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "Unknown error" << std::endl;
        return 2;
    }
}
