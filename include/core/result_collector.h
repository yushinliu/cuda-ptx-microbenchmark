#pragma once

#include "benchmark_runner.h"
#include <map>
#include <vector>
#include <string>

namespace cpm {

/**
 * @brief Collector for organizing and analyzing benchmark results
 */
class ResultCollector {
public:
    ResultCollector();
    ~ResultCollector();

    /**
     * @brief Add a result to the collector
     */
    void add_result(const BenchmarkResult& result);

    /**
     * @brief Get all results for a specific benchmark name
     */
    const std::vector<BenchmarkResult>& get_results(const std::string& name) const;

    /**
     * @brief Get total number of results stored
     */
    size_t get_result_count() const;

    /**
     * @brief Clear all results
     */
    void clear();

    /**
     * @brief Check if results exist for a given name
     */
    bool has_result(const std::string& name) const;

    /**
     * @brief Calculate average value for a benchmark
     */
    float get_average_value(const std::string& name) const;

    /**
     * @brief Get all unique benchmark names
     */
    std::vector<std::string> get_benchmark_names() const;

    /**
     * @brief Export results to JSON format
     */
    std::string export_to_json() const;

private:
    std::map<std::string, std::vector<BenchmarkResult>> results_;
};

}  // namespace cpm
