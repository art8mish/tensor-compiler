cmake_minimum_required(VERSION 3.16)
set(KERNELS_TARGET kernels)

if(KERNELS_AS_SUBDIRECTORY)
    set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
else()
    project(${KERNELS_TARGET} LANGUAGES CXX)
    set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

set(KERNELS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(KERNELS_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

function(add_kernels_lib target)
    add_library(${target} STATIC ${ARGN})
    target_include_directories(${target} PUBLIC ${INCLUDE_DIR})
    target_compile_features(${target} PUBLIC cxx_std_20)
endfunction()

function(add_kernels_exec target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${INCLUDE_DIR})
    target_compile_features(${target} PRIVATE cxx_std_20)
endfunction()

add_kernels_lib(${KERNELS_TARGET}
    ${KERNELS_SRC_DIR}/matmul.cpp
    ${KERNELS_SRC_DIR}/conv.cpp
    ${KERNELS_SRC_DIR}/runners.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    target_compile_options(${KERNELS_TARGET} PRIVATE -mavx2 -mfma)
endif()

include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
FetchContent_MakeAvailable(googlebenchmark)

set(BENCHMARK_TARGET kbenchmark)
add_kernels_exec(${BENCHMARK_TARGET} ${KERNELS_SRC_DIR}/benchmark.cpp)
target_link_libraries(${BENCHMARK_TARGET} ${KERNELS_TARGET} benchmark::benchmark)


option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(${KERNELS_TESTS_DIR})
endif()
