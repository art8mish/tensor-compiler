enable_testing() 
find_package(GTest REQUIRED) 
include(GoogleTest)

function(add_proj_test target) 
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
        GTest::gtest_main 
        ${GRAPHVIZ_LIBRARIES} 
        ${ONNX_LIBRARY} 
        ${ONNX_PROTO_LIBRARY} 
        ${Protobuf_LIBRARIES}
        ${MLIR_LIBS}
        ${llvm_libs}
    )
    target_compile_definitions(${target} PRIVATE ONNX_ML)
    target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS})
    gtest_discover_tests(${target})
endfunction()

add_proj_test(tests ${TESTS_DIR}/tests.cpp)