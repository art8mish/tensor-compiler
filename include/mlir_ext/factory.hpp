#pragma once

#include <memory>
#include <vector>
#include <string>

#include "cgraph.hpp"
#include "mlir_ext/graph.hpp"
#include "mlir_ext/builder.hpp"


// MLIR includes
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace tensor_compiler {

class MLIRComputeGraphFactory {
public:
    static std::unique_ptr<MLIRComputeGraph> create(const ComputeGraph &graph, 
                                                    const std::string &name = "main_module") {
        auto context = std::make_unique<mlir::MLIRContext>();
        context->getOrLoadDialect<mlir::linalg::LinalgDialect>();
        context->getOrLoadDialect<mlir::arith::ArithDialect>();
        context->getOrLoadDialect<mlir::tensor::TensorDialect>();
        context->getOrLoadDialect<mlir::math::MathDialect>();
        context->getOrLoadDialect<mlir::func::FuncDialect>();

        MLIRBuilder builder(context.get());
        mlir::Location loc = builder.builder_.getUnknownLoc();
        
        auto module = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(loc, name));

        std::vector<mlir::Type> input_types;
        std::vector<const Node*> args;
        
        for (auto* node : graph.nodes()) {
            if (node->type() == NodeType::TENSOR) {
                auto* tnode = static_cast<const TensorNode *>(node);
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

        auto* entry_block = func_op.addEntryBlock();
        builder.builder_.setInsertionPointToStart(entry_block);

        auto size = static_cast<unsigned int>(args.size());
        for (unsigned int i = 0; i < size; ++i)
            builder.value_map_[args[i]] = entry_block->getArgument(i);

        const Node* last_node = nullptr;
        for (auto* node : graph.nodes()) {
            builder.process_node(node);
            last_node = node;
        }
        if (last_node) {
            builder.finalize(last_node);
        }
        
        return std::make_unique<MLIRComputeGraph>(std::move(context), std::move(module));
    }
};

} // namespace tensor_compiler