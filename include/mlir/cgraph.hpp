#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotBufferize.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"


#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"

namespace tensor_compiler {

class MLIRComputeGraph {
    std::unique_ptr<mlir::MLIRContext> context_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;

public:
    MLIRComputeGraph(std::unique_ptr<mlir::MLIRContext> ctx, 
                     mlir::OwningOpRef<mlir::ModuleOp> mod)
        : context_(std::move(ctx)), module_(std::move(mod)) {}

    std::unique_ptr<llvm::Module> generateLLVMIR(llvm::LLVMContext &llvm_context) {
        mlir::PassManager pm(context_.get());

        mlir::bufferization::OneShotBufferizationOptions options;
        options.allowReturnAllocs = true; 
        options.bufferizeFunctionBoundaries = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));

        pm.addPass(mlir::createConvertLinalgToLoopsPass()); 
        pm.addPass(mlir::createConvertSCFToCFPass());

        // LLVM Dialect
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createMemRefToLLVMConversionPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        if (mlir::failed(pm.run(*module_)))
            throw std::runtime_error("MLIR lowering failed");

        mlir::registerLLVMDialectTranslation(*context_);
        auto llvmModule = mlir::translateModuleToLLVMIR(*module_, llvm_context);
        
        return llvmModule;
    }

    void dump() { module_->dump(); }
};

} // namespace tensor_compiler