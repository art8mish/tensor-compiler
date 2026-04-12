cmake_minimum_required(VERSION 3.16)
set(TARGET tcompiler)
project(${TARGET} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRAPHVIZ REQUIRED IMPORTED_TARGET libgvc libcgraph)

include_directories(${GRAPHVIZ_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
find_library(ONNX_LIBRARY NAMES onnx REQUIRED)
find_path(ONNX_INCLUDE_DIR NAMES onnx/onnx_pb.h REQUIRED)
find_library(ONNX_PROTO_LIBRARY NAMES onnx_proto REQUIRED)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

llvm_map_components_to_libnames(llvm_libs
    Core
    Support
    Target
    TargetParser
    AsmPrinter
    MC
    Passes
    native
    nativecodegen
)
set(MLIR_LIBS
    MLIRAffineToStandard
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRVectorToLLVM
    MLIRVectorToLLVMPass
    MLIRVectorToSCF
    MLIRMathToLLVM
    MLIRLLVMCommonConversion
    MLIRIndexToLLVM
    
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    
    MLIRReconcileUnrealizedCasts
    MLIRVectorTransforms
    MLIRArithTransforms
    MLIRTensorTransforms
    MLIRFuncTransforms
    MLIRSCFTransforms
    MLIRLinalgTransforms
    MLIRMemRefTransforms
    MLIRBufferizationTransforms
    MLIRTransforms
    MLIRPass
    
    MLIRVectorDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRLinalgUtils
    MLIRArithDialect
    MLIRMathDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRBufferizationDialect
    MLIRLLVMDialect
    MLIRIndexDialect
    MLIRIR

    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)


function(add_exec target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE 
        ${INCLUDE_DIR} 
        ${GRAPHVIZ_INCLUDE_DIRS} 
        ${ONNX_INCLUDE_DIR} 
        ${Protobuf_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
    )
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target} PRIVATE 
        ${GRAPHVIZ_LIBRARIES} 
        ${ONNX_LIBRARY} 
        ${ONNX_PROTO_LIBRARY} 
        ${Protobuf_LIBRARIES}
        ${MLIR_LIBS}
        ${llvm_libs}
    )
    target_compile_definitions(${target} PRIVATE ONNX_ML)
    target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS})
endfunction()

function(add_warnings target)
    target_compile_options(${target} PRIVATE 
        # -Werror
        -Wall
        -Wextra
        -Wpedantic
        -Wcast-align
        -Wcast-qual
        -Wconversion
        -Wold-style-cast
        -Wredundant-decls
        -Wsign-conversion
        -Wsign-promo
    )
endfunction()

add_exec(${TARGET} ${SRC_DIR}/main.cpp)

set(MODEL_ONNX_PATH "${CMAKE_SOURCE_DIR}/models/model.onnx")
set(MODEL_OBJ_PATH "${CMAKE_CURRENT_BINARY_DIR}/model.o")
option(BUILD_MODEL_DRIVER "Build ${MODEL_DRIVER} (links ${MODEL_OBJ_PATH}; requires ${MODEL_ONNX_PATH})" OFF)

if(BUILD_MODEL_DRIVER)
    
    set(MODEL_DRIVER model_driver)
    set(MODEL_OBJ model_obj)

    add_custom_command(
        OUTPUT ${MODEL_OBJ_PATH}
        COMMAND $<TARGET_FILE:${TARGET}> ${MODEL_ONNX_PATH} -o ${MODEL_OBJ_PATH}
        DEPENDS ${TARGET}
        COMMENT "Compiling ${MODEL_OBJ_PATH}..."
        VERBATIM
    )

    add_custom_target(${MODEL_OBJ} DEPENDS ${MODEL_OBJ_PATH})

    add_executable(${MODEL_DRIVER} ${SRC_DIR}/driver.cpp)
    target_include_directories(${MODEL_DRIVER} PRIVATE
        ${INCLUDE_DIR}
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
    )
    target_compile_features(${MODEL_DRIVER} PRIVATE cxx_std_20)
    target_link_libraries(${MODEL_DRIVER} PRIVATE ${MODEL_OBJ_PATH})
    target_compile_definitions(${MODEL_DRIVER} PRIVATE ${LLVM_DEFINITIONS})
    add_dependencies(${MODEL_DRIVER} ${MODEL_OBJ})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_warnings(${TARGET})
endif()

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    add_subdirectory(${TESTS_DIR})
endif()