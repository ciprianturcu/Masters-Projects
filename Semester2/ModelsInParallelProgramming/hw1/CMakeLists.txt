cmake_minimum_required(VERSION 3.30)
project(hw1)

set(CMAKE_CXX_STANDARD 20)

add_executable(hw1 main.cpp)
add_executable(matrixGeneration matrixGeneration.cpp)

# Set the output directory for runtime files (executables)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(sequential1 sequential1.cpp)
