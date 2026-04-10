#pragma once

#include <memory>
#include <string>

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace tensor_compiler {


class MLIRComputeGraph {
public:
    MLIRComputeGraph(std::unique_ptr<mlir::MLIRContext>  ctx,
                     mlir::OwningOpRef<mlir::ModuleOp>   mod)
        : context_(std::move(ctx)), module_(std::move(mod)) {}

    MLIRComputeGraph(const MLIRComputeGraph &)            = delete;
    MLIRComputeGraph &operator=(const MLIRComputeGraph &) = delete;
    MLIRComputeGraph(MLIRComputeGraph &&)                 = default;
    MLIRComputeGraph &operator=(MLIRComputeGraph &&)      = default;
    ~MLIRComputeGraph() = default;

    mlir::MLIRContext &context() { return *context_; }
    const mlir::MLIRContext &context() const { return *context_; }

    mlir::ModuleOp module() { return module_.get(); }
    const mlir::ModuleOp module() const { return module_.get(); }

    mlir::OwningOpRef<mlir::ModuleOp> &owning_module() { return module_; }

    void dump() const { module_.get().dump(); }

private:
    std::unique_ptr<mlir::MLIRContext>  context_;
    mlir::OwningOpRef<mlir::ModuleOp>  module_;
};

} // namespace tensor_compiler