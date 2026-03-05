cmake_minimum_required(VERSION 3.16)
set(TARGET tensor-compiler)
project(${TARGET} LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GRAPHVIZ REQUIRED libgvc libcgraph)

include_directories(${GRAPHVIZ_INCLUDE_DIRS})

find_package(onnxruntime REQUIRED)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

function(add_exec target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${INCLUDE_DIR} ${GRAPHVIZ_INCLUDE_DIRS})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target} PRIVATE ${GRAPHVIZ_LIBRARIES})
    target_link_libraries(${target} PRIVATE onnxruntime::onnxruntime)
endfunction()

function(add_warnings target)
    target_compile_options(${target} PRIVATE 
        -Werror
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


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_warnings(${TARGET})
endif()

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    add_subdirectory(${TESTS_DIR})
endif()