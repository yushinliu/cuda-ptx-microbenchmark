# Tests CMakeLists.txt

# Test fixtures library
add_library(test_fixtures STATIC
    fixtures/gpu_test_fixture.cpp
    fixtures/benchmark_fixture.cpp
)
target_include_directories(test_fixtures PUBLIC
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_fixtures PUBLIC
    GTest::gtest
    cpm_core
    CUDA::cudart
)
set_target_properties(test_fixtures PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Unit tests
add_executable(unit_tests
    unit/test_timer.cu
    unit/test_result_collector.cpp
    unit/test_ptx_assembler.cpp
)
target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(unit_tests PRIVATE
    test_fixtures
    cpm_kernels
    GTest::gtest_main
)
set_target_properties(unit_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
add_test(NAME UnitTests COMMAND unit_tests)

# Integration tests
add_executable(integration_tests
    integration/test_l1_cache.cu
    integration/test_l2_cache.cu
    integration/test_ptx_instructions.cu
    integration/test_memory_bandwidth.cu
)
target_include_directories(integration_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(integration_tests PRIVATE
    test_fixtures
    cpm_kernels
    GTest::gtest_main
)
set_target_properties(integration_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
add_test(NAME IntegrationTests COMMAND integration_tests)

# E2E tests
add_executable(e2e_tests
    e2e/test_full_benchmark_suite.cu
    e2e/test_report_generation.cu
)
target_include_directories(e2e_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(e2e_tests PRIVATE
    test_fixtures
    cpm_kernels
    GTest::gtest_main
)
set_target_properties(e2e_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
add_test(NAME E2ETests COMMAND e2e_tests)

# Microbenchmark tests (new TDD-based tests)
# Include kernel sources directly to ensure proper CUDA linking
add_executable(microbench_tests
    microbench/test_memory_bandwidth.cu
    microbench/test_memory_latency.cu
    microbench/test_shared_memory.cu
    microbench/test_instruction_latency.cu
    microbench/test_instruction_throughput.cu
    microbench/test_warp_scheduler.cu
    microbench/test_integer_instructions.cu
    microbench/test_double_precision.cu
    microbench/test_l2_cache.cu
    microbench/test_ada_specific.cu
    microbench/test_tensor_cores.cu
    microbench/test_shared_memory_banks.cu
    # Kernel sources for proper device linking
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/integer_instructions.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/double_precision.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/l2_cache.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/ada_specific.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/tensor_cores.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/shared_memory_banks.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/instruction_latency.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/instruction_throughput.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/memory_bandwidth.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/memory_latency.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/microbench/shared_memory.cu
)
target_include_directories(microbench_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(microbench_tests PRIVATE
    test_fixtures
    GTest::gtest_main
)
set_target_properties(microbench_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_test(NAME MicrobenchTests COMMAND microbench_tests)

# Test labels for selective running
set_tests_properties(UnitTests PROPERTIES LABELS "unit")
set_tests_properties(IntegrationTests PROPERTIES LABELS "integration")
set_tests_properties(E2ETests PROPERTIES LABELS "e2e")
set_tests_properties(MicrobenchTests PROPERTIES LABELS "microbench")
