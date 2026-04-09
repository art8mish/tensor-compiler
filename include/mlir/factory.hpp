#pragma once

#include <memory>
#include <vector>
#include <string>

#include "mlir_compute_graph.hpp"
#include "mlir_builder.hpp" // Предполагаем, что MLIRBuilder вынесен в этот файл
#include "nodes/compute_graph.hpp"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace tensor_compiler {

class MLIRComputeGraphFactory {
public:
    static std::unique_ptr<MLIRComputeGraph> create(const ComputeGraph &graph, 
                                                    const std::string &name = "main_module") {
        auto context = std::make_unique<mlir::MLIRContext>();
        context->getOrLoadDialect<mlir::linalg::LinalgDialect,
                                  mlir::arith::ArithDialect,
                                  mlir::tensor::TensorDialect,
                                  mlir::math::MathDialect,
                                  mlir::func::FuncDialect>();

        MLIRBuilder builder(context.get());
        mlir::Location loc = builder.builder_.getUnknownLoc();
        
        auto module = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(loc, name));

        std::vector<mlir::Type> input_types;
        std::vector<const Node*> args;
        
        for (auto* node : graph.nodes()) {
            if (node->type() == NodeType::TENSOR) {
                auto* tnode = static_cast<const TensorNode<>*>(node);
                const TensorPtr &tensor = tnode->tensor();
                if (!tnode->input() && tensor && tensor->empty()) {
                    input_types.push_back(builder.get_tensor_type(tnode));
                    args.push_back(node);
                }
            }
        }

        builder.builder_.setInsertionPointToEnd(module->getBody());
        auto func_type = builder.builder_.getFunctionType(input_types, {});
        auto func_op = builder.builder_.create<mlir::func::FuncOp>(loc, "forward", func_type);
        
        auto* entry_block = func_op.addEntryBlock();
        builder.builder_.setInsertionPointToStart(entry_block);

        for (size_t i = 0; i < args.size(); ++i)
            builder.value_map_[args[i]] = entryBlock->getArgument(i);

        for (auto* node : graph.nodes())
            builder.process_node(node);
        
        return std::make_unique<MLIRComputeGraph>(std::move(context), std::move(module));
    }
};

} // namespace tensor_compiler