#include "core/benchmark_runner.h"
#include <map>

namespace cpm {

class BenchmarkRunner::Impl {
public:
    BenchmarkConfig config;
    std::map<std::string, std::function<BenchmarkResult()>> benchmarks;
};

BenchmarkRunner::BenchmarkRunner(const BenchmarkConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

BenchmarkRunner::~BenchmarkRunner() = default;

void BenchmarkRunner::register_benchmark(const std::string& name,
                                         std::function<BenchmarkResult()> func) {
    impl_->benchmarks[name] = std::move(func);
}

BenchmarkResult BenchmarkRunner::run_single(const std::string& name) {
    auto it = impl_->benchmarks.find(name);
    if (it == impl_->benchmarks.end()) {
        BenchmarkResult result;
        result.name = name;
        result.status = BenchmarkStatus::kError;
        result.error_message = "Benchmark not found: " + name;
        return result;
    }

    return it->second();
}

std::vector<BenchmarkResult> BenchmarkRunner::run_all() {
    std::vector<BenchmarkResult> results;
    results.reserve(impl_->benchmarks.size());

    for (const auto& [name, func] : impl_->benchmarks) {
        results.push_back(func());
    }

    return results;
}

std::vector<std::string> BenchmarkRunner::get_benchmark_names() const {
    std::vector<std::string> names;
    names.reserve(impl_->benchmarks.size());

    for (const auto& [name, _] : impl_->benchmarks) {
        names.push_back(name);
    }

    return names;
}

}  // namespace cpm
