cmake_minimum_required(VERSION 3.25)

project(edge-ml-runtime-lab
        VERSION 0.1.0
        DESCRIPTION "A C++ tensor/autograd engine, a quantized deployment pipeline and a graph optimizer"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Written for clangd and the IDEs; costs nothing and saves every future setup step.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TINYTENSOR_BUILD_TESTS "Build the test suite" ON)
option(TINYTENSOR_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

add_subdirectory(tinytensor)

if(TINYTENSOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
