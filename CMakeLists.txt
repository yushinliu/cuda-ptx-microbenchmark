cmake_minimum_required(VERSION 3.18)
project(CudaPtxMicrobenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)  # Ada Lovelace (RTX 4070)

# Options
option(ENABLE_TESTING "Enable test suite" ON)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_PROFILING "Enable profiling support" OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -Xcompiler -Wextra")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Coverage flags
if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler --coverage")
endif()

# Find packages
find_package(CUDAToolkit REQUIRED)

if(ENABLE_TESTING)
    find_package(GTest REQUIRED)
    enable_testing()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Core library (CPU code)
add_library(cpm_core STATIC
    src/core/timer.cpp
    src/core/benchmark_runner.cpp
    src/core/result_collector.cpp
)
target_include_directories(cpm_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# CUDA kernels library
add_library(cpm_kernels STATIC
    src/kernels/memory/l1_cache.cu
    src/kernels/memory/l2_cache.cu
    src/kernels/memory/global_memory.cu
    src/kernels/ptx/arithmetic.cu
    src/kernels/ptx/memory_ptx.cu
    src/kernels/ptx/synchronization.cu
    src/kernels/microbench/integer_instructions.cu
    src/kernels/microbench/double_precision.cu
    src/kernels/microbench/l2_cache.cu
    src/kernels/microbench/ada_specific.cu
    src/kernels/microbench/tensor_cores.cu
    src/kernels/microbench/shared_memory_banks.cu
)
target_link_libraries(cpm_kernels PUBLIC cpm_core)
set_target_properties(cpm_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Main executable
add_executable(cpm src/main.cpp)
target_link_libraries(cpm PRIVATE cpm_core cpm_kernels)

# Testing
if(ENABLE_TESTING)
    add_subdirectory(tests)
endif()

# Install targets
install(TARGETS cpm DESTINATION bin)
install(TARGETS cpm_core cpm_kernels DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
