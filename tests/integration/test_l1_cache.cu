#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "fixtures/benchmark_fixture.h"
#include "kernels/memory/l1_cache.h"

namespace cpm {

class L1CacheBenchmarkTest : public BenchmarkTestFixture {
protected:
    static constexpr size_t kL1Size = 128 * 1024;  // 128KB L1 cache
    static constexpr size_t kCacheLine = 128;       // 128B cache line
};

TEST_F(L1CacheBenchmarkTest, test_l1_hit_rate_with_sequential_access) {
    // Given: Data size fitting in L1
    const size_t data_size = kL1Size / 2;
    L1CacheBenchmark benchmark(data_size);

    // When: Execute sequential access test
    auto result = benchmark.run_sequential_access();

    // Then: Hit rate should be high
    EXPECT_EQ(result.status, L1CacheStatus::kSuccess);
    EXPECT_GT(result.hit_rate, 0.95f);
    EXPECT_GT(result.bandwidth_gbps, 1000.0f);  // L1 bandwidth should be high
    EXPECT_GT(result.elapsed_ms, 0.0f);
    EXPECT_GT(result.bytes_processed, 0);
}

TEST_F(L1CacheBenchmarkTest, test_l1_miss_with_random_access) {
    // Given: Data size exceeding L1
    const size_t data_size = kL1Size * 4;
    L1CacheBenchmark benchmark(data_size);

    // When: Execute random access test
    auto result = benchmark.run_random_access();

    // Then: Should have significant misses
    EXPECT_EQ(result.status, L1CacheStatus::kSuccess);
    EXPECT_LT(result.hit_rate, 0.5f);
    EXPECT_LT(result.bandwidth_gbps, 500.0f);  // Bandwidth should drop
}

TEST_F(L1CacheBenchmarkTest, test_bandwidth_within_theoretical_limits) {
    // Given: RTX 4070 theoretical L1 bandwidth ~10+ TB/s
    const float theoretical_max = 12000.0f;  // GB/s
    L1CacheBenchmark benchmark(kL1Size / 2);

    // When: Measure bandwidth
    auto result = benchmark.run_sequential_access();

    // Then: Measured should not exceed theoretical peak
    EXPECT_LT(result.bandwidth_gbps, theoretical_max * 1.1f);
    EXPECT_GT(result.bandwidth_gbps, 0.0f);
}

// Edge case tests
TEST_F(L1CacheBenchmarkTest, test_empty_data_handles_gracefully) {
    L1CacheBenchmark benchmark(0);
    auto result = benchmark.run_sequential_access();

    EXPECT_EQ(result.status, L1CacheStatus::kSkipped);
}

TEST_F(L1CacheBenchmarkTest, test_small_data_size) {
    // Test with very small data (single cache line)
    L1CacheBenchmark benchmark(kCacheLine);
    auto result = benchmark.run_sequential_access();

    EXPECT_EQ(result.status, L1CacheStatus::kSuccess);
    EXPECT_GT(result.hit_rate, 0.99f);  // Should be nearly 100%
}

TEST_F(L1CacheBenchmarkTest, test_exact_l1_size) {
    // Test with exactly L1 size
    L1CacheBenchmark benchmark(kL1Size);
    auto result = benchmark.run_sequential_access();

    EXPECT_EQ(result.status, L1CacheStatus::kSuccess);
    EXPECT_GT(result.bandwidth_gbps, 0.0f);
}

TEST_F(L1CacheBenchmarkTest, test_unaligned_access) {
    // Test with non-aligned data
    L1CacheBenchmark benchmark(kL1Size / 2, /* alignment */ 1);
    auto result = benchmark.run_sequential_access();

    EXPECT_EQ(result.status, L1CacheStatus::kSuccess);
    EXPECT_GT(result.bandwidth_gbps, 0.0f);
}

// Statistical consistency test
TEST_F(L1CacheBenchmarkTest, test_results_are_consistent_across_runs) {
    const size_t data_size = kL1Size / 2;
    L1CacheBenchmark benchmark(data_size);

    std::vector<float> bandwidths;
    std::vector<float> hit_rates;

    // Run multiple times
    for (int i = 0; i < 5; ++i) {
        auto result = benchmark.run_sequential_access();
        EXPECT_EQ(result.status, L1CacheStatus::kSuccess);
        bandwidths.push_back(result.bandwidth_gbps);
        hit_rates.push_back(result.hit_rate);
    }

    // Results should be consistent within 10%
    EXPECT_TRUE(results_are_consistent(bandwidths, 0.10f));
    EXPECT_TRUE(results_are_consistent(hit_rates, 0.05f));
}

// Random access consistency test
TEST_F(L1CacheBenchmarkTest, test_random_access_consistency) {
    const size_t data_size = kL1Size;
    L1CacheBenchmark benchmark(data_size);

    std::vector<float> bandwidths;

    // Run multiple times
    for (int i = 0; i < 5; ++i) {
        auto result = benchmark.run_random_access();
        EXPECT_EQ(result.status, L1CacheStatus::kSuccess);
        bandwidths.push_back(result.bandwidth_gbps);
    }

    // Results should be consistent within 15% (random has more variance)
    EXPECT_TRUE(results_are_consistent(bandwidths, 0.15f));
}

// Test that sequential bandwidth exceeds random bandwidth
TEST_F(L1CacheBenchmarkTest, test_sequential_faster_than_random) {
    const size_t data_size = kL1Size / 2;

    L1CacheBenchmark seq_benchmark(data_size);
    auto seq_result = seq_benchmark.run_sequential_access();

    L1CacheBenchmark rand_benchmark(data_size);
    auto rand_result = rand_benchmark.run_random_access();

    EXPECT_EQ(seq_result.status, L1CacheStatus::kSuccess);
    EXPECT_EQ(rand_result.status, L1CacheStatus::kSuccess);

    // Sequential should have higher bandwidth than random
    EXPECT_GT(seq_result.bandwidth_gbps, rand_result.bandwidth_gbps);

    // Sequential should have higher hit rate
    EXPECT_GT(seq_result.hit_rate, rand_result.hit_rate);
}

// Parameterized test for different data sizes
class L1CacheSizeTest : public BenchmarkTestFixture,
                        public ::testing::WithParamInterface<size_t> {};

TEST_P(L1CacheSizeTest, test_various_data_sizes) {
    size_t data_size = GetParam();
    L1CacheBenchmark benchmark(data_size);

    auto result = benchmark.run_sequential_access();

    EXPECT_EQ(result.status, L1CacheStatus::kSuccess);
    EXPECT_GT(result.bandwidth_gbps, 0.0f);
    EXPECT_GE(result.hit_rate, 0.0f);
    EXPECT_LE(result.hit_rate, 1.0f);
}

INSTANTIATE_TEST_SUITE_P(
    DataSizes,
    L1CacheSizeTest,
    ::testing::Values(
        1024,        // 1 KB
        64 * 1024,   // 64 KB
        128 * 1024,  // 128 KB (L1 size)
        256 * 1024,  // 256 KB (2x L1)
        1024 * 1024  // 1 MB
    )
);

// Error handling tests
TEST_F(L1CacheBenchmarkTest, test_move_constructor) {
    L1CacheBenchmark benchmark1(kL1Size / 2);
    auto result1 = benchmark1.run_sequential_access();
    EXPECT_EQ(result1.status, L1CacheStatus::kSuccess);

    // Move construct
    L1CacheBenchmark benchmark2(std::move(benchmark1));
    auto result2 = benchmark2.run_sequential_access();
    EXPECT_EQ(result2.status, L1CacheStatus::kSuccess);
}

TEST_F(L1CacheBenchmarkTest, test_move_assignment) {
    L1CacheBenchmark benchmark1(kL1Size / 2);
    auto result1 = benchmark1.run_sequential_access();
    EXPECT_EQ(result1.status, L1CacheStatus::kSuccess);

    L1CacheBenchmark benchmark2(kL1Size / 4);
    benchmark2 = std::move(benchmark1);

    auto result2 = benchmark2.run_sequential_access();
    EXPECT_EQ(result2.status, L1CacheStatus::kSuccess);
}

}  // namespace cpm
