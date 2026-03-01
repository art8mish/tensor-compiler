enable_testing() 
find_package(GTest REQUIRED) 
include(GoogleTest)

function(add_proj_test target) 
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${GRAPHVIZ_INCLUDE_DIRS})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target} PRIVATE GTest::gtest_main ${GRAPHVIZ_LIBRARIES})
    gtest_discover_tests(${target})
endfunction()

add_proj_test(tests ${TESTS_DIR}/tests.cpp)