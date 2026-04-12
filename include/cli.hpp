#pragma once

#include "mlir_ext/compiler.hpp"

#include <iosfwd>
#include <optional>
#include <string>

namespace tensor_compiler {

struct Cli {
    bool help = false;
    bool parse_error = false;
    std::string error_message;

    std::optional<std::string> input_onnx;
    std::optional<std::string> out_path;
    std::optional<std::string> out_ll;

    std::optional<std::string> out_graph;
    std::optional<std::string> out_asm;
    std::optional<std::string> assembly;

    LLVMOptions llvm_opts;

    static std::string process_arg_val(int argc, char **argv, int &i, std::string def) {
        auto out = def;
        if (i + 1 < argc && argv[i + 1][0] != '-') {
            out = argv[i + 1];
            ++i;
        }
        return out;
    }

    Cli(int argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            const char *arg = argv[i];
            if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
                help = true;
                return;
            }
            if (std::strcmp(arg, "-S") == 0) {
                out_asm = process_arg_val(argc, argv, i, "results/out.s");
                continue;
            }
            if (std::strcmp(arg, "-G") == 0) {
                out_graph = process_arg_val(argc, argv, i,"results/graph.png");
                continue;
            }
            if (std::strcmp(arg, "-l") == 0) {
                out_ll = process_arg_val(argc, argv, i,"results/out.ll");
                continue;
            }
            if (std::strcmp(arg, "-o") == 0) {
                out_path = process_arg_val(argc, argv, i,"results/out.o");
                continue;
            }
            if (const char *v = eq_value(arg, "--llvm-triple")) {
                llvm_opts.triple = v;
                continue;
            }
            if (const char *v = eq_value(arg, "--llvm-cpu")) {
                llvm_opts.cpu = v;
                continue;
            }
            if (const char *v = eq_value(arg, "--llvm-features")) {
                llvm_opts.features = v;
                continue;
            }
            if (const char *v = eq_value(arg, "--reloc")) {
                if (std::strcmp(v, "pic") == 0)
                    llvm_opts.reloc = LLVMReloc::PIC;
                else if (std::strcmp(v, "static") == 0)
                    llvm_opts.reloc = LLVMReloc::Static;
                else if (std::strcmp(v, "default") == 0)
                    llvm_opts.reloc = LLVMReloc::Default;
                else
                    throw std::invalid_argument("Invalid --reloc value: " + std::string(v));
                continue;
            }
            if (arg[0] == '-')
                throw std::invalid_argument("Unknown option: " + std::string(arg));
            input_onnx = arg;
        }
        if (!input_onnx)
            throw std::invalid_argument("Expected <model.onnx>");

        if (!out_graph && !out_path && !out_asm && !out_ll)
            throw std::invalid_argument("Expected [-o <output.o>|-S <output.s>|-G");
    }

    const char *eq_value(const char *arg, const char *prefix) {
        const size_t n = std::strlen(prefix);
        if (std::strncmp(arg, prefix, n) != 0)
            return nullptr;
        if (arg[n] != '=')
            return nullptr;
        return arg + n + 1;
    }

    // static std::string derive_assembly_path(const std::string &output_path) {
    //     const auto pos = output_path.rfind('.');
    //     if (pos == std::string::npos || pos == 0)
    //         return output_path + ".s";
    //     return output_path.substr(0, pos) + ".s";
    // }

    // static OutputFormat infer_output_format(const std::string &output_path, const char *legacy_fmt_opt) {
    //     if (legacy_fmt_opt) {
    //         if (std::strcmp(legacy_fmt_opt, "object") == 0)
    //             return OutputFormat::Object;
    //         return OutputFormat::Assembly;
    //     }
    //     if (output_path.size() >= 2 && output_path.compare(output_path.size() - 2, 2, ".o") == 0)
    //         return OutputFormat::Object;
    //     return OutputFormat::Assembly;
    // }

    static void print_help(std::ostream &os, const char *argv0) {
        os << "Usage: " << argv0
        << " [options] <model.onnx> (-o [path.o] | -S [path.s] | -l [path.ll] | -G [path.png]) \n"
            "\n"
            "Options:\n"
            "  -h, --help              Show this help\n"
            "  -o [path.o]             Create object file. Default: results/out.o\n"
            "  -S [path.s]             Create assembly file. Default: results/out.s\n"
            "  -l [path.ll]            LLVM IR output. Default: results/out.ll\n"
            "  -G [path.png]           Compute graph image (Graphviz PNG). Default: results/graph.png\n"
            "  --llvm-triple=TRIPLE    LLVM target triple\n"
            "  --llvm-cpu=CPU          LLVM target CPU\n"
            "  --llvm-features=STR     LLVM subtarget features string\n"
            "  --reloc=pic|static|default  Relocation model (default: pic)\n"
            "\n"
            "Examples:\n"
            "  " << argv0 << " model.onnx -o out.o\n"
            "  " << argv0 << " -S model.onnx -o out.o   # writes results/out.s\n";
    }
};

} // namespace tensor_compiler
