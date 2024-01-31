cmake_minimum_required(VERSION 3.21)
project(tensor)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(include)
add_subdirectory(src)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)

if (BUILD_TESTS)
    enable_testing()
    message("Building tests")
    add_subdirectory(tests)
endif()

if (BUILD_EXAMPLES)
    message("Building examples")
    add_subdirectory(examples)
endif()

