cmake_minimum_required(VERSION 3.16.3)

project(MOCHA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# ================================================================
# Add CUDA include dir so device_launch_parameters.h is found
include_directories(/usr/local/cuda-11.8/include)

# ================================================================
# Add your own project header folders
# Adjust to your real paths
# ================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/Tool
    ${CMAKE_CURRENT_SOURCE_DIR}/HostDeviceData
    ${CMAKE_CURRENT_SOURCE_DIR}/Driver
)

# ================================================================
# Automatically find all .cu and .cpp files in your structure
# ================================================================

# Get all CUDA source files in Kernel/
file(GLOB KERNEL_CU ${CMAKE_CURRENT_SOURCE_DIR}/Kernel/*.cu)

# Get all C++ source files in Tool/
file(GLOB TOOL_CPP ${CMAKE_CURRENT_SOURCE_DIR}/Tool/*.cpp)

# Get main Problem.cpp in Driver/
file(GLOB DRIVER_CPP ${CMAKE_CURRENT_SOURCE_DIR}/Driver/*.cpp)

# Combine all sources
set(SOURCES
    ${KERNEL_CU}
    ${TOOL_CPP}
    ${DRIVER_CPP}
)

# ================================================================
# Create executable
# ================================================================
add_executable(MOCHA ${SOURCES})

# For multiple .cu files, enable separable compilation
set_target_properties(MOCHA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Example: enable fast math if you want
target_compile_options(MOCHA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
