#pragma once

#include "gpu_test_fixture.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace cpm {

/**
 * @brief Base test fixture for benchmark tests
 *
 * Extends GpuTestFixture with benchmark-specific utilities
 * for result validation and statistical analysis.
 */
class BenchmarkTestFixture : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();

        // Check for RTX 4070 or compatible Ada architecture
        if (device_props_.major != 8 || device_props_.minor != 9) {
            std::cout << "Warning: Running on " << device_props_.name
                      << " (compute " << device_props_.major << "."
                      << device_props_.minor << "), expected Ada (8.9)"
                      << std::endl;
        }

        // Set deterministic stack size
        cudaDeviceSetLimit(cudaLimitStackSize, 8192);
    }

    /**
     * @brief Calculate mean of a vector
     */
    template<typename T>
    T calculate_mean(const std::vector<T>& values) {
        if (values.empty()) return T(0);
        T sum = std::accumulate(values.begin(), values.end(), T(0));
        return sum / values.size();
    }

    /**
     * @brief Calculate median of a vector
     */
    template<typename T>
    T calculate_median(std::vector<T> values) {
        if (values.empty()) return T(0);
        size_t n = values.size();
        std::sort(values.begin(), values.end());
        if (n % 2 == 0) {
            return (values[n/2 - 1] + values[n/2]) / 2;
        } else {
            return values[n/2];
        }
    }

    /**
     * @brief Calculate standard deviation
     */
    template<typename T>
    T calculate_stddev(const std::vector<T>& values, T mean) {
        if (values.size() < 2) return T(0);
        T sum_sq_diff = 0;
        for (const auto& v : values) {
            T diff = v - mean;
            sum_sq_diff += diff * diff;
        }
        return std::sqrt(sum_sq_diff / (values.size() - 1));
    }

    /**
     * @brief Check if results are consistent within tolerance
     *
     * @param results Vector of measurements
     * @param tolerance Relative tolerance (e.g., 0.05 for 5%)
     * @return true if all results within tolerance of mean
     */
    template<typename T>
    bool results_are_consistent(const std::vector<T>& results, float tolerance) {
        if (results.size() < 2) return true;

        T mean = calculate_mean(results);
        if (mean == T(0)) return false;

        for (const auto& r : results) {
            if (std::abs(r - mean) > std::abs(mean) * tolerance) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Check if value is within theoretical limits
     */
    bool is_within_theoretical_limits(float measured, float theoretical_max,
                                       float tolerance = 0.1f) {
        return measured <= theoretical_max * (1.0f + tolerance);
    }

    // RTX 4070 specifications
    static constexpr float k4070MemoryBandwidth = 504.0f;  // GB/s
    static constexpr float k4070L1Bandwidth = 12000.0f;     // GB/s (approximate)
    static constexpr float k4070L2Bandwidth = 2000.0f;      // GB/s (approximate)
    static constexpr size_t kL1CacheSize = 128 * 1024;      // 128 KB per SM
    static constexpr size_t kL2CacheSize = 36 * 1024 * 1024; // 36 MB
};

}  // namespace cpm
