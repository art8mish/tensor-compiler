#pragma once

#include <memory>
#include <optional>
#include <mlir/Dialect/Math/IR/Math.h>
#include <string>
#include <system_error>

#include "mlir_ext/graph.hpp"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

namespace tensor_compiler {

enum class OutputFormat { Assembly, Object };

enum class LLVMReloc { Default, PIC, Static };

struct LLVMOptions {
    std::optional<std::string> triple;
    std::optional<std::string> cpu;
    std::optional<std::string> features;
    LLVMReloc reloc = LLVMReloc::Default;
};

class MLIRCompiler {
public:
    MLIRCompiler() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
    }

    auto compile(MLIRComputeGraph &graph, const std::string &out_path, OutputFormat format,
                   const LLVMOptions &llvm_opts = {},
                   const std::optional<std::string> &assembly_path = std::nullopt, const std::optional<std::string> &ll_path = std::nullopt) {
        mlir::DialectRegistry registry;
        registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                        mlir::bufferization::BufferizationDialect, mlir::cf::ControlFlowDialect,
                        mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::LLVM::LLVMDialect,
                        mlir::math::MathDialect, mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                        mlir::tensor::TensorDialect>();

        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);

        auto &ctx = graph.context();
        ctx.appendDialectRegistry(registry);
        ctx.loadAllAvailableDialects();

        mlir::PassManager pm(&ctx);

        mlir::bufferization::OneShotBufferizePassOptions buf_opts;
        buf_opts.bufferizeFunctionBoundaries = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(buf_opts));

        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());
        pm.addPass(mlir::memref::createExpandStridedMetadataPass());

        pm.addPass(mlir::createConvertIndexToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertMathToLLVMPass());
        pm.addPass(mlir::createConvertVectorToLLVMPass());

        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());

        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        if (mlir::failed(pm.run(graph.module())))
            return false;
        if (mlir::failed(mlir::verify(graph.module())))
            return false;

        llvm::LLVMContext llvm_ctx;
        auto llvm_mod = mlir::translateModuleToLLVMIR(graph.module(), llvm_ctx);
        if (!llvm_mod)
            return false;
        if (llvm::verifyModule(*llvm_mod, &llvm::errs()))
            return false;
        return llvm_ctx
    }

private:
    static llvm::Reloc::Model to_llvm_reloc(LLVMReloc r) {
        switch (r) {
        case LLVMReloc::Static:
            return llvm::Reloc::Static;
        case LLVMReloc::PIC:
        case LLVMReloc::Default:
        default:
            return llvm::Reloc::PIC_;
        }
    }

    static bool emit_file(llvm::Module &mod, const std::string &path, OutputFormat format,
                   const LLVMOptions &opts) {
        std::string triple_str =
            opts.triple.value_or(llvm::sys::getDefaultTargetTriple());
        llvm::Triple triple(triple_str);

        std::string err;
        const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, err);
        if (!target)
            return false;

        std::string cpu = opts.cpu.value_or(llvm::sys::getHostCPUName().str());
        std::string feat = opts.features.value_or("");

        llvm::TargetOptions opt;
        auto reloc_model = to_llvm_reloc(opts.reloc);

        std::unique_ptr<llvm::TargetMachine> tm(
            target->createTargetMachine(triple, cpu, feat, opt, reloc_model));

        if (!tm)
            return false;

        mod.setDataLayout(tm->createDataLayout());
        mod.setTargetTriple(triple);

        std::error_code ec;
        llvm::raw_fd_ostream dest(path, ec);
        if (ec)
            return false;

        llvm::legacy::PassManager cg_pm;
        auto file_type = (format == OutputFormat::Assembly) ? llvm::CodeGenFileType::AssemblyFile
                                                            : llvm::CodeGenFileType::ObjectFile;

        if (tm->addPassesToEmitFile(cg_pm, dest, nullptr, file_type))
            return false;

        cg_pm.run(mod);
        dest.flush();
        return true;
    }
};

} // namespace tensor_compiler
