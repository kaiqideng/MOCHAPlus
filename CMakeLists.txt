cmake_minimum_required(VERSION 3.16.3)

project(MOCHA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# ================================================================
# CUDA Toolkit include path
# ================================================================
include_directories(/usr/local/cuda-11.8/include)

# ================================================================
# Add the root path for public headers if needed
# ================================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add Driver folder for headers if you need
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Driver)

# ================================================================
# Collect ALL .cpp and .cu files in root path (not Driver/)
# ================================================================
file(GLOB ROOT_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB ROOT_CU ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# ================================================================
# Collect ALL .cpp files in Driver/ (each is a main)
# ================================================================
file(GLOB DRIVER_MAIN_CPP ${CMAKE_CURRENT_SOURCE_DIR}/Driver/*.cpp)

# ================================================================
# For each Driver main, build an executable and link ROOT files
# ================================================================
foreach(MAIN_CPP ${DRIVER_MAIN_CPP})

    get_filename_component(MAIN_NAME ${MAIN_CPP} NAME_WE)

    add_executable(${MAIN_NAME}
        ${MAIN_CPP}  # This Driver file (has main)
        ${ROOT_CPP}  # Common .cpp modules in root
        ${ROOT_CU}   # Common .cu modules in root
    )

    set_target_properties(${MAIN_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

endforeach()
