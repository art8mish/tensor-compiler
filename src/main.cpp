

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
#include <llvm/Support/raw_os_ostream.h> 
#include <stdexcept>
#include <string>

namespace tc=tensor_compiler;

#ifndef NDEBUG
#endif

int main(int argc, char **argv) {
    try {
        tc::Cli cli(argc, argv);
        if (cli.help) {
            tc::Cli::print_help(std::cout, argv[0]);
            return 0;
        }

        auto start_time = std::clock();
        std::cout << "Creating computation graph...\n";
        auto graph = tc::ComputeGraphFactory::from_onnx(*cli.input_onnx);

        if (cli.out_graph) {
            std::filesystem::path gp(*cli.out_graph);
            if (!gp.has_parent_path())
                std::filesystem::create_directories(gp.parent_path());
            tc::draw_compute_graph(*graph, *cli.out_graph);
            std::cout << "Graph written: " << *cli.out_graph << std::endl;
        }

        if (!cli.out_path && !cli.out_asm && !cli.out_ll)
            return 0;

        std::cout << "Converting graph to MLIR...\n";
        auto mlir_g = tc::MLIRComputeGraphFactory::create(*graph, "mlir_graph");
    
        
        std::cout << "Converting MLIR to LLVM IR...\n";
        tc::MLIRCompiler compiler;
        llvm::LLVMContext llvm_ctx;
        auto llvm_mod = compiler.build_llvm(*mlir_g, llvm_ctx);
        if (!llvm_mod) {
            std::cout << "Compilation failed\n";
            return 1;
        }

        if (cli.out_ll) {
            std::ofstream out_ll(*cli.out_ll);
            if (!out_ll.is_open())
                throw std::invalid_argument("File opening is failed: " + *cli.out_ll);
            llvm::raw_os_ostream ll_stream(out_ll);
            llvm_mod->print(ll_stream, nullptr);
            ll_stream.flush();
            std::cout << "LLVM IR file writed: " << *cli.out_ll << std::endl;
        }

        if (cli.out_asm) {
            std::cout << "Compiling LLVM IR to assembler...\n";
            compiler.compile(*llvm_mod, *cli.out_asm, tc::OutputFormat::Assembly, cli.llvm_opts);
            std::cout << "Assembly file writed: " << *cli.out_path << std::endl;
        }
        
        if (cli.out_path) {
            std::cout << "Compiling LLVM IR to object...\n";
            compiler.compile(*llvm_mod, *cli.out_path, tc::OutputFormat::Object, cli.llvm_opts);
            std::cout << "Object file writed: " << *cli.out_path << std::endl;
        }

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
