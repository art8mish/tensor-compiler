#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cgraph.hpp"
#include "mlir_ext/builder.hpp"
#include "mlir_ext/graph.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>

#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>

#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

namespace tensor_compiler {

class MLIRComputeGraphFactory {
public:
    static std::unique_ptr<MLIRComputeGraph> create(const ComputeGraph &graph,
                                                    const std::string &name = "main_module") {
        auto context = make_mlir_context();

        MLIRBuilder builder(context.get());
        mlir::Location loc = builder.builder_.getUnknownLoc();

        auto module = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(loc, name));

        std::vector<mlir::Type> input_types;
        std::vector<const Node *> args;

        for (auto *node : graph.nodes()) {
            if (node->type() == NodeType::TENSOR) {
                auto *tnode = static_cast<const TensorNode *>(node);
                const Tensor *tensor = tnode->tensor();
                if (!tnode->input() && tensor && tensor->empty()) {
                    input_types.push_back(builder.get_tensor_type(tnode));
                    args.push_back(node);
                }
            }
        }

        builder.builder_.setInsertionPointToEnd(module->getBody());
        auto func_type = builder.builder_.getFunctionType(input_types, {});
        auto func_op = mlir::func::FuncOp::create(builder.builder_, loc, "forward", func_type);
        func_op->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(context.get()));

        auto *entry_block = func_op.addEntryBlock();
        builder.builder_.setInsertionPointToStart(entry_block);

        auto size = static_cast<unsigned int>(args.size());
        for (unsigned int i = 0; i < size; ++i)
            builder.value_map_[args[i]] = entry_block->getArgument(i);

        const Node *last_node = nullptr;
        for (auto *node : graph.nodes()) {
            builder.process_node(node);
            last_node = node;
        }
        if (last_node) {
            builder.finalize(last_node);
        }

        return std::make_unique<MLIRComputeGraph>(std::move(context), std::move(module));
    }

private:
    static void register_dialects(mlir::DialectRegistry &registry) {
        registry
            .insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect, mlir::cf::ControlFlowDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::LLVM::LLVMDialect,
                    mlir::math::MathDialect, mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::tensor::TensorDialect>();

        mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
    }

    static std::unique_ptr<mlir::MLIRContext> make_mlir_context() {
        mlir::DialectRegistry registry;
        register_dialects(registry);
        auto context = std::make_unique<mlir::MLIRContext>();
        context->appendDialectRegistry(registry);
        context->loadAllAvailableDialects();
        return context;
    }
};

} // namespace tensor_compiler
