#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "fixtures/benchmark_fixture.h"
#include "kernels/memory/l1_cache.h"
#include "kernels/memory/l2_cache.h"

namespace cpm {

class L2CacheBenchmarkTest : public BenchmarkTestFixture {
protected:
    // RTX 4070 L2 cache specs
    static constexpr size_t kL2Size = 36 * 1024 * 1024;  // 36 MB
    static constexpr size_t kCacheLine = 128;             // 128B cache line
    static constexpr float kTheoreticalBandwidth = 2000.0f;  // ~2 TB/s
};

TEST_F(L2CacheBenchmarkTest, test_l2_hit_rate_with_sequential_access) {
    // Given: Data size fitting well in L2
    const size_t data_size = kL2Size / 4;
    L2CacheBenchmark benchmark(data_size);

    // When: Execute sequential access test
    auto result = benchmark.run_sequential_access();

    // Then: Hit rate should be high
    EXPECT_EQ(result.status, L2CacheStatus::kSuccess);
    EXPECT_GT(result.hit_rate, 0.90f);
    EXPECT_GT(result.bandwidth_gbps, 1000.0f);
    EXPECT_GT(result.elapsed_ms, 0.0f);
    EXPECT_GT(result.bytes_processed, 0);
}

TEST_F(L2CacheBenchmarkTest, test_l2_miss_with_large_data) {
    // Given: Data size exceeding L2
    const size_t data_size = kL2Size * 2;
    L2CacheBenchmark benchmark(data_size);

    // When: Execute sequential access (streaming)
    auto result = benchmark.run_sequential_access();

    // Then: Should have some misses due to capacity
    EXPECT_EQ(result.status, L2CacheStatus::kSuccess);
    EXPECT_LT(result.hit_rate, 1.0f);
}

TEST_F(L2CacheBenchmarkTest, test_bandwidth_within_theoretical_limits) {
    // Given: RTX 4070 theoretical L2 bandwidth ~2 TB/s
    L2CacheBenchmark benchmark(kL2Size / 2);

    // When: Measure bandwidth
    auto result = benchmark.run_sequential_access();

    // Then: Measured should not exceed theoretical peak
    EXPECT_LT(result.bandwidth_gbps, kTheoreticalBandwidth * 1.1f);
    EXPECT_GT(result.bandwidth_gbps, 0.0f);
}

TEST_F(L2CacheBenchmarkTest, test_stride_access_patterns) {
    // Given: Different stride patterns
    const size_t data_size = kL2Size / 2;

    // Small stride should have good locality
    L2CacheBenchmark benchmark_small(data_size);
    auto result_small = benchmark_small.run_stride_access(32);
    EXPECT_EQ(result_small.status, L2CacheStatus::kSuccess);
    EXPECT_GT(result_small.hit_rate, 0.7f);
    EXPECT_EQ(result_small.stride, 32);

    // Large stride should have poor locality
    L2CacheBenchmark benchmark_large(data_size);
    auto result_large = benchmark_large.run_stride_access(1024);
    EXPECT_EQ(result_large.status, L2CacheStatus::kSuccess);
    EXPECT_LT(result_large.hit_rate, 0.5f);
    EXPECT_EQ(result_large.stride, 1024);
}

// Edge cases
TEST_F(L2CacheBenchmarkTest, test_empty_data) {
    L2CacheBenchmark benchmark(0);
    auto result = benchmark.run_sequential_access();
    EXPECT_EQ(result.status, L2CacheStatus::kSkipped);
}

TEST_F(L2CacheBenchmarkTest, test_zero_stride) {
    L2CacheBenchmark benchmark(kL2Size / 2);
    auto result = benchmark.run_stride_access(0);
    EXPECT_EQ(result.status, L2CacheStatus::kInvalidStride);
}

TEST_F(L2CacheBenchmarkTest, test_cache_line_size_data) {
    // Test with exactly one cache line
    L2CacheBenchmark benchmark(kCacheLine);
    auto result = benchmark.run_sequential_access();

    EXPECT_EQ(result.status, L2CacheStatus::kSuccess);
    EXPECT_GT(result.hit_rate, 0.99f);
}

TEST_F(L2CacheBenchmarkTest, test_exact_l2_size) {
    // Test with exactly L2 size
    L2CacheBenchmark benchmark(kL2Size);
    auto result = benchmark.run_sequential_access();

    EXPECT_EQ(result.status, L2CacheStatus::kSuccess);
    EXPECT_GT(result.bandwidth_gbps, 0.0f);
}

// Random access test
TEST_F(L2CacheBenchmarkTest, test_random_access) {
    // Given: Data size fitting in L2
    const size_t data_size = kL2Size / 2;
    L2CacheBenchmark benchmark(data_size);

    // When: Execute random access test
    auto result = benchmark.run_random_access();

    // Then: Should complete successfully
    EXPECT_EQ(result.status, L2CacheStatus::kSuccess);
    EXPECT_GT(result.bandwidth_gbps, 0.0f);
    EXPECT_GE(result.hit_rate, 0.0f);
    EXPECT_LE(result.hit_rate, 1.0f);
}

// Statistical consistency
TEST_F(L2CacheBenchmarkTest, test_results_consistency) {
    const size_t data_size = kL2Size / 2;
    L2CacheBenchmark benchmark(data_size);

    std::vector<float> bandwidths;
    std::vector<float> hit_rates;

    for (int i = 0; i < 5; ++i) {
        auto result = benchmark.run_sequential_access();
        EXPECT_EQ(result.status, L2CacheStatus::kSuccess);
        bandwidths.push_back(result.bandwidth_gbps);
        hit_rates.push_back(result.hit_rate);
    }

    // Results should be consistent within 15%
    EXPECT_TRUE(results_are_consistent(bandwidths, 0.15f));
    EXPECT_TRUE(results_are_consistent(hit_rates, 0.10f));
}

// Stride consistency test
TEST_F(L2CacheBenchmarkTest, test_stride_consistency) {
    const size_t data_size = kL2Size / 2;
    const size_t stride = 256;

    L2CacheBenchmark benchmark(data_size);

    std::vector<float> bandwidths;

    for (int i = 0; i < 5; ++i) {
        auto result = benchmark.run_stride_access(stride);
        EXPECT_EQ(result.status, L2CacheStatus::kSuccess);
        EXPECT_EQ(result.stride, stride);
        bandwidths.push_back(result.bandwidth_gbps);
    }

    EXPECT_TRUE(results_are_consistent(bandwidths, 0.15f));
}

// Compare L1 vs L2 bandwidth (L1 should be faster for small data)
TEST_F(L2CacheBenchmarkTest, test_l1_faster_than_l2_for_small_data) {
    const size_t small_data = 64 * 1024;  // 64KB - fits in L1

    // L1 benchmark
    L1CacheBenchmark l1_benchmark(small_data);
    auto l1_result = l1_benchmark.run_sequential_access();

    // L2 benchmark
    L2CacheBenchmark l2_benchmark(small_data);
    auto l2_result = l2_benchmark.run_sequential_access();

    EXPECT_EQ(l1_result.status, L1CacheStatus::kSuccess);
    EXPECT_EQ(l2_result.status, L2CacheStatus::kSuccess);

    // L1 should have higher bandwidth for small data
    EXPECT_GT(l1_result.bandwidth_gbps, l2_result.bandwidth_gbps);
}

// Parameterized test for different data sizes
class L2CacheSizeTest : public BenchmarkTestFixture,
                        public ::testing::WithParamInterface<size_t> {};

TEST_P(L2CacheSizeTest, test_various_data_sizes) {
    size_t data_size = GetParam();
    L2CacheBenchmark benchmark(data_size);

    auto result = benchmark.run_sequential_access();

    if (data_size == 0) {
        EXPECT_EQ(result.status, L2CacheStatus::kSkipped);
    } else {
        EXPECT_EQ(result.status, L2CacheStatus::kSuccess);
        EXPECT_GT(result.bandwidth_gbps, 0.0f);
        EXPECT_GE(result.hit_rate, 0.0f);
        EXPECT_LE(result.hit_rate, 1.0f);
    }
}

INSTANTIATE_TEST_SUITE_P(
    DataSizes,
    L2CacheSizeTest,
    ::testing::Values(
        0,                    // Edge case: empty
        1024,                 // 1 KB
        64 * 1024,            // 64 KB
        1024 * 1024,          // 1 MB
        16 * 1024 * 1024,     // 16 MB (half L2)
        36 * 1024 * 1024,     // 36 MB (full L2)
        72 * 1024 * 1024      // 72 MB (2x L2)
    )
);

// Parameterized test for different strides
class L2CacheStrideTest : public BenchmarkTestFixture,
                          public ::testing::WithParamInterface<size_t> {};

TEST_P(L2CacheStrideTest, test_various_strides) {
    size_t stride = GetParam();
    const size_t kL2Size = 36 * 1024 * 1024;  // 36 MB - RTX 4070 L2 size
    const size_t data_size = kL2Size / 4;

    L2CacheBenchmark benchmark(data_size);
    auto result = benchmark.run_stride_access(stride);

    EXPECT_EQ(result.status, L2CacheStatus::kSuccess);
    EXPECT_GT(result.bandwidth_gbps, 0.0f);
    EXPECT_EQ(result.stride, stride);
}

INSTANTIATE_TEST_SUITE_P(
    Strides,
    L2CacheStrideTest,
    ::testing::Values(
        1,      // Sequential
        8,      // Small stride
        32,     // Within cache line
        64,     // Two cache lines
        256,    // Large stride
        1024    // Very large stride
    )
);

// Move semantics tests
TEST_F(L2CacheBenchmarkTest, test_move_constructor) {
    L2CacheBenchmark benchmark1(kL2Size / 2);
    auto result1 = benchmark1.run_sequential_access();
    EXPECT_EQ(result1.status, L2CacheStatus::kSuccess);

    // Move construct
    L2CacheBenchmark benchmark2(std::move(benchmark1));
    auto result2 = benchmark2.run_sequential_access();
    EXPECT_EQ(result2.status, L2CacheStatus::kSuccess);
}

TEST_F(L2CacheBenchmarkTest, test_move_assignment) {
    L2CacheBenchmark benchmark1(kL2Size / 2);
    auto result1 = benchmark1.run_sequential_access();
    EXPECT_EQ(result1.status, L2CacheStatus::kSuccess);

    L2CacheBenchmark benchmark2(kL2Size / 4);
    benchmark2 = std::move(benchmark1);

    auto result2 = benchmark2.run_sequential_access();
    EXPECT_EQ(result2.status, L2CacheStatus::kSuccess);
}

}  // namespace cpm
