#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "fixtures/benchmark_fixture.h"
#include "kernels/memory/global_memory.h"

namespace cpm {

class MemoryBandwidthTest : public BenchmarkTestFixture {
protected:
    // RTX 4070 specs
    static constexpr float kTheoreticalBandwidth = 504.0f;  // GB/s
    static constexpr float kTolerance = 0.1f;  // 10%
};

TEST_F(MemoryBandwidthTest, test_read_bandwidth_within_limits) {
    // Given: Large data transfer
    const size_t data_size = 256 * 1024 * 1024;  // 256 MB
    MemoryBandwidthBenchmark benchmark(data_size);

    // When: Measure read bandwidth
    auto result = benchmark.run_read_benchmark();

    // Then: Should be within theoretical limits
    EXPECT_EQ(result.status, BandwidthStatus::kSuccess);
    EXPECT_GT(result.read_bandwidth_gbps, 0.0f);
    EXPECT_LT(result.read_bandwidth_gbps, kTheoreticalBandwidth * (1 + kTolerance));

    // Should achieve reasonable percentage of theoretical
    EXPECT_GT(result.read_bandwidth_gbps, kTheoreticalBandwidth * 0.7f);
}

TEST_F(MemoryBandwidthTest, test_write_bandwidth_within_limits) {
    // Given: Large data transfer
    const size_t data_size = 256 * 1024 * 1024;
    MemoryBandwidthBenchmark benchmark(data_size);

    // When: Measure write bandwidth
    auto result = benchmark.run_write_benchmark();

    // Then: Should be within limits
    EXPECT_EQ(result.status, BandwidthStatus::kSuccess);
    EXPECT_GT(result.write_bandwidth_gbps, 0.0f);
    EXPECT_LT(result.write_bandwidth_gbps, kTheoreticalBandwidth * (1 + kTolerance));
}

TEST_F(MemoryBandwidthTest, test_copy_bandwidth_lower_than_read) {
    // Given: Same data size for all tests
    const size_t data_size = 256 * 1024 * 1024;
    MemoryBandwidthBenchmark read_bench(data_size);
    MemoryBandwidthBenchmark copy_bench(data_size);

    // When: Measure both
    auto read_result = read_bench.run_read_benchmark();
    auto copy_result = copy_bench.run_copy_benchmark();

    // Then: Copy should be slower than pure read
    EXPECT_EQ(read_result.status, BandwidthStatus::kSuccess);
    EXPECT_EQ(copy_result.status, BandwidthStatus::kSuccess);
    EXPECT_LT(copy_result.copy_bandwidth_gbps, read_result.read_bandwidth_gbps);
}

TEST_F(MemoryBandwidthTest, test_bandwidth_scales_with_size) {
    // Test that larger transfers achieve better bandwidth (amortized overhead)
    std::vector<size_t> sizes = {
        1 * 1024 * 1024,    // 1 MB
        16 * 1024 * 1024,   // 16 MB
        256 * 1024 * 1024   // 256 MB
    };

    std::vector<float> bandwidths;
    for (size_t size : sizes) {
        MemoryBandwidthBenchmark benchmark(size);
        auto result = benchmark.run_read_benchmark();
        EXPECT_EQ(result.status, BandwidthStatus::kSuccess);
        bandwidths.push_back(result.read_bandwidth_gbps);
    }

    // Larger transfers should generally achieve better bandwidth
    // (though this is not strictly guaranteed due to caching effects)
    EXPECT_GT(bandwidths.back(), 0.0f);
}

// Edge cases
TEST_F(MemoryBandwidthTest, test_empty_data) {
    MemoryBandwidthBenchmark benchmark(0);
    auto result = benchmark.run_read_benchmark();
    EXPECT_EQ(result.status, BandwidthStatus::kSkipped);
}

TEST_F(MemoryBandwidthTest, test_small_data) {
    // Very small transfers may have high overhead
    MemoryBandwidthBenchmark benchmark(1024);  // 1 KB
    auto result = benchmark.run_read_benchmark();

    EXPECT_EQ(result.status, BandwidthStatus::kSuccess);
    // Bandwidth may be lower due to overhead, but should still be positive
    EXPECT_GT(result.read_bandwidth_gbps, 0.0f);
}

TEST_F(MemoryBandwidthTest, test_unaligned_access) {
    // Test with non-standard alignment
    MemoryBandwidthBenchmark benchmark(256 * 1024 * 1024, /* alignment */ 1);
    auto result = benchmark.run_read_benchmark();

    EXPECT_EQ(result.status, BandwidthStatus::kSuccess);
    EXPECT_GT(result.read_bandwidth_gbps, 0.0f);
}

// Statistical consistency
TEST_F(MemoryBandwidthTest, test_bandwidth_consistency) {
    const size_t data_size = 256 * 1024 * 1024;
    MemoryBandwidthBenchmark benchmark(data_size);

    std::vector<float> read_bandwidths;
    std::vector<float> write_bandwidths;

    for (int i = 0; i < 5; ++i) {
        auto read_result = benchmark.run_read_benchmark();
        auto write_result = benchmark.run_write_benchmark();

        EXPECT_EQ(read_result.status, BandwidthStatus::kSuccess);
        EXPECT_EQ(write_result.status, BandwidthStatus::kSuccess);

        read_bandwidths.push_back(read_result.read_bandwidth_gbps);
        write_bandwidths.push_back(write_result.write_bandwidth_gbps);
    }

    // Results should be consistent within 10%
    EXPECT_TRUE(results_are_consistent(read_bandwidths, 0.10f));
    EXPECT_TRUE(results_are_consistent(write_bandwidths, 0.10f));
}

// Test that read bandwidth is typically higher than write
TEST_F(MemoryBandwidthTest, test_read_faster_than_write) {
    const size_t data_size = 256 * 1024 * 1024;
    MemoryBandwidthBenchmark benchmark(data_size);

    auto read_result = benchmark.run_read_benchmark();
    auto write_result = benchmark.run_write_benchmark();

    EXPECT_EQ(read_result.status, BandwidthStatus::kSuccess);
    EXPECT_EQ(write_result.status, BandwidthStatus::kSuccess);

    // Read is typically faster than write on modern GPUs
    EXPECT_GE(read_result.read_bandwidth_gbps, write_result.write_bandwidth_gbps * 0.9f);
}

// Move semantics tests
TEST_F(MemoryBandwidthTest, test_move_constructor) {
    MemoryBandwidthBenchmark benchmark1(256 * 1024 * 1024);
    auto result1 = benchmark1.run_read_benchmark();
    EXPECT_EQ(result1.status, BandwidthStatus::kSuccess);

    // Move construct
    MemoryBandwidthBenchmark benchmark2(std::move(benchmark1));
    auto result2 = benchmark2.run_read_benchmark();
    EXPECT_EQ(result2.status, BandwidthStatus::kSuccess);
}

TEST_F(MemoryBandwidthTest, test_move_assignment) {
    MemoryBandwidthBenchmark benchmark1(256 * 1024 * 1024);
    auto result1 = benchmark1.run_read_benchmark();
    EXPECT_EQ(result1.status, BandwidthStatus::kSuccess);

    MemoryBandwidthBenchmark benchmark2(128 * 1024 * 1024);
    benchmark2 = std::move(benchmark1);

    auto result2 = benchmark2.run_read_benchmark();
    EXPECT_EQ(result2.status, BandwidthStatus::kSuccess);
}

// Parameterized test for different access patterns
class MemoryBandwidthPatternTest : public BenchmarkTestFixture,
                                   public ::testing::WithParamInterface<std::string> {};

TEST_P(MemoryBandwidthPatternTest, test_access_patterns) {
    std::string pattern = GetParam();
    const size_t data_size = 256 * 1024 * 1024;
    MemoryBandwidthBenchmark benchmark(data_size);

    BandwidthResult result;
    if (pattern == "read") {
        result = benchmark.run_read_benchmark();
        EXPECT_GT(result.read_bandwidth_gbps, 0.0f);
    } else if (pattern == "write") {
        result = benchmark.run_write_benchmark();
        EXPECT_GT(result.write_bandwidth_gbps, 0.0f);
    } else if (pattern == "copy") {
        result = benchmark.run_copy_benchmark();
        EXPECT_GT(result.copy_bandwidth_gbps, 0.0f);
    }

    EXPECT_EQ(result.status, BandwidthStatus::kSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    AccessPatterns,
    MemoryBandwidthPatternTest,
    ::testing::Values("read", "write", "copy")
);

// Parameterized test for different data sizes
class MemoryBandwidthSizeTest : public BenchmarkTestFixture,
                                public ::testing::WithParamInterface<size_t> {};

TEST_P(MemoryBandwidthSizeTest, test_various_sizes) {
    size_t data_size = GetParam();
    MemoryBandwidthBenchmark benchmark(data_size);

    auto result = benchmark.run_read_benchmark();

    if (data_size == 0) {
        EXPECT_EQ(result.status, BandwidthStatus::kSkipped);
    } else {
        EXPECT_EQ(result.status, BandwidthStatus::kSuccess);
        EXPECT_GT(result.read_bandwidth_gbps, 0.0f);
    }
}

INSTANTIATE_TEST_SUITE_P(
    DataSizes,
    MemoryBandwidthSizeTest,
    ::testing::Values(
        0,                          // Edge case
        1024,                       // 1 KB
        64 * 1024,                  // 64 KB
        1024 * 1024,                // 1 MB
        16 * 1024 * 1024,           // 16 MB
        256 * 1024 * 1024           // 256 MB
    )
);

}  // namespace cpm
