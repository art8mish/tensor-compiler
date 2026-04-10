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

    

    Cli(int argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            const char *arg = argv[i];
            if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
                help = true;
                return;
            }
            if (std::strcmp(arg, "-S") == 0) {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    out_asm = argv[i + 1];
                    ++i;
                }
                else
                    out_asm = std::string("results/out.s");
                continue;
            }
            if (std::strcmp(arg, "-G") == 0) {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    out_graph = argv[i + 1];
                    ++i;
                } else
                    out_graph = std::string("results/graph.png");
                continue;
            }
            if (std::strcmp(arg, "-l") == 0) {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    out_ll = argv[i + 1];
                    ++i;
                }
                else
                    out_ll = std::string("results/out.ll");
                continue;
            }
            if (std::strcmp(arg, "-o") == 0) {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    out_path = argv[i + 1];
                    ++i;
                }
                else
                    throw std::invalid_argument("Invalid -o value");
                continue;
            }


            if (const char *v = eq_value(arg, "--graph")) {
                out_graph = v;
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

        if (!out_path)
            throw std::invalid_argument("Expected -o <output.o>");
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
        << " [options] <model.onnx> -o <output.o>\n"
            "\n"
            "Options:\n"
            "  -h, --help              Show this help\n"
            "  -S [path]               Create assembly file. Default: results/out.s\n"
            "  -G [path]               Compute graph image (Graphviz PNG). Default: results/graph.png\n"
            "  -l [path]               LLVM IR output (.ll). Default: results/out.ll\n"
            "  --graph=path            Same as -G with explicit path\n"
            "  --llvm-triple=TRIPLE    LLVM target triple\n"
            "  --llvm-cpu=CPU          LLVM target CPU\n"
            "  --llvm-features=STR     LLVM subtarget features string\n"
            "  --reloc=pic|static|default  Relocation model (default: pic)\n"
            "\n"
            "Examples:\n"
            "  " << argv0 << " model.onnx -o out.o\n"
            "  " << argv0 << " -S model.onnx -o out.o   # also writes results/out.s\n";
    }
};

} // namespace tensor_compiler
