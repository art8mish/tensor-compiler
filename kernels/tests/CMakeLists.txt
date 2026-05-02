enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

function(add_kernels_test target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${INCLUDE_DIR})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target} PRIVATE
        GTest::gtest_main
        ${KERNELS_TARGET}
    )
    gtest_discover_tests(${target})
endfunction()

add_kernels_test(kernel_tests ${KERNELS_TESTS_DIR}/kernels_tests.cpp)
