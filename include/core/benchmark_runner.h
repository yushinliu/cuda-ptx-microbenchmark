#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace cpm {

/**
 * @brief Status of a benchmark run
 */
enum class BenchmarkStatus {
    kSuccess = 0,
    kSkipped = 1,
    kError = 2,
    kTimeout = 3
};

/**
 * @brief Result of a benchmark execution
 */
struct BenchmarkResult {
    std::string name;
    float value;
    std::string unit;
    float min;
    float max;
    float mean;
    float stddev;
    int iterations;
    float elapsed_time_ms;
    BenchmarkStatus status;
    std::string error_message;
};

/**
 * @brief Configuration for benchmark execution
 */
struct BenchmarkConfig {
    int warmup_iterations = 10;
    int measurement_iterations = 100;
    float timeout_seconds = 30.0f;
    bool verify_results = true;
    bool print_progress = false;
};

/**
 * @brief Runner for executing benchmarks
 */
class BenchmarkRunner {
public:
    explicit BenchmarkRunner(const BenchmarkConfig& config = {});
    ~BenchmarkRunner();

    /**
     * @brief Register a benchmark function
     */
    void register_benchmark(const std::string& name,
                           std::function<BenchmarkResult()> func);

    /**
     * @brief Run a single benchmark by name
     */
    BenchmarkResult run_single(const std::string& name);

    /**
     * @brief Run all registered benchmarks
     */
    std::vector<BenchmarkResult> run_all();

    /**
     * @brief Get list of registered benchmark names
     */
    std::vector<std::string> get_benchmark_names() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpm
