#include "core/result_collector.h"
#include <numeric>
#include <sstream>

namespace cpm {

ResultCollector::ResultCollector() = default;
ResultCollector::~ResultCollector() = default;

void ResultCollector::add_result(const BenchmarkResult& result) {
    results_[result.name].push_back(result);
}

const std::vector<BenchmarkResult>& ResultCollector::get_results(
    const std::string& name) const {
    static const std::vector<BenchmarkResult> empty;
    auto it = results_.find(name);
    if (it != results_.end()) {
        return it->second;
    }
    return empty;
}

size_t ResultCollector::get_result_count() const {
    size_t count = 0;
    for (const auto& [_, results] : results_) {
        count += results.size();
    }
    return count;
}

void ResultCollector::clear() {
    results_.clear();
}

bool ResultCollector::has_result(const std::string& name) const {
    return results_.find(name) != results_.end();
}

float ResultCollector::get_average_value(const std::string& name) const {
    auto it = results_.find(name);
    if (it == results_.end() || it->second.empty()) {
        return 0.0f;
    }

    float sum = std::accumulate(
        it->second.begin(), it->second.end(), 0.0f,
        [](float acc, const BenchmarkResult& r) { return acc + r.value; });

    return sum / static_cast<float>(it->second.size());
}

std::vector<std::string> ResultCollector::get_benchmark_names() const {
    std::vector<std::string> names;
    names.reserve(results_.size());

    for (const auto& [name, _] : results_) {
        names.push_back(name);
    }

    return names;
}

std::string ResultCollector::export_to_json() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"benchmarks\": [\n";

    bool first = true;
    for (const auto& [name, results] : results_) {
        for (const auto& result : results) {
            if (!first) oss << ",\n";
            first = false;

            oss << "    {\n";
            oss << "      \"name\": \"" << result.name << "\",\n";
            oss << "      \"value\": " << result.value << ",\n";
            oss << "      \"unit\": \"" << result.unit << "\",\n";
            oss << "      \"mean\": " << result.mean << ",\n";
            oss << "      \"stddev\": " << result.stddev << "\n";
            oss << "    }";
        }
    }

    oss << "\n  ]\n";
    oss << "}\n";

    return oss.str();
}

}  // namespace cpm
