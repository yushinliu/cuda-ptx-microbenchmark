#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <map>

namespace cpm {

// Minimal ResultCollector implementation for testing
struct BenchmarkResult {
    std::string name;
    float value;
    std::string unit;
    float min;
    float max;
    float mean;
    float stddev;
    int iterations;
    bool valid;
};

class ResultCollector {
public:
    void addResult(const BenchmarkResult& result) {
        results_[result.name].push_back(result);
    }

    const std::vector<BenchmarkResult>& getResults(const std::string& name) const {
        static const std::vector<BenchmarkResult> empty;
        auto it = results_.find(name);
        if (it != results_.end()) {
            return it->second;
        }
        return empty;
    }

    size_t getResultCount() const {
        size_t count = 0;
        for (const auto& pair : results_) {
            count += pair.second.size();
        }
        return count;
    }

    void clear() {
        results_.clear();
    }

    bool hasResult(const std::string& name) const {
        return results_.find(name) != results_.end();
    }

    float getAverageValue(const std::string& name) const {
        auto it = results_.find(name);
        if (it == results_.end() || it->second.empty()) {
            return 0.0f;
        }

        float sum = 0.0f;
        for (const auto& r : it->second) {
            sum += r.value;
        }
        return sum / it->second.size();
    }

private:
    std::map<std::string, std::vector<BenchmarkResult>> results_;
};

// Test fixture
class ResultCollectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        collector_ = std::make_unique<ResultCollector>();
    }

    void TearDown() override {
        collector_.reset();
    }

    std::unique_ptr<ResultCollector> collector_;
};

// RED: Write failing test first
TEST_F(ResultCollectorTest, test_add_single_result) {
    // Given: Empty collector
    ASSERT_EQ(collector_->getResultCount(), 0);

    // When: Add a result
    BenchmarkResult result;
    result.name = "test_benchmark";
    result.value = 100.0f;
    result.unit = "GB/s";
    result.valid = true;

    collector_->addResult(result);

    // Then: Result is stored
    EXPECT_EQ(collector_->getResultCount(), 1);
    EXPECT_TRUE(collector_->hasResult("test_benchmark"));
}

TEST_F(ResultCollectorTest, test_add_multiple_results_same_name) {
    // Given: Multiple results with same name
    for (int i = 0; i < 5; ++i) {
        BenchmarkResult result;
        result.name = "repeated_benchmark";
        result.value = 100.0f + i * 10.0f;
        result.unit = "ns";
        result.valid = true;
        collector_->addResult(result);
    }

    // Then: All results stored under same name
    auto results = collector_->getResults("repeated_benchmark");
    EXPECT_EQ(results.size(), 5);
}

TEST_F(ResultCollectorTest, test_add_results_different_names) {
    // Given: Results with different names
    BenchmarkResult result1;
    result1.name = "bandwidth_test";
    result1.value = 500.0f;
    result1.unit = "GB/s";
    result1.valid = true;

    BenchmarkResult result2;
    result2.name = "latency_test";
    result2.value = 10.0f;
    result2.unit = "ns";
    result2.valid = true;

    // When: Add both results
    collector_->addResult(result1);
    collector_->addResult(result2);

    // Then: Both stored correctly
    EXPECT_EQ(collector_->getResultCount(), 2);
    EXPECT_TRUE(collector_->hasResult("bandwidth_test"));
    EXPECT_TRUE(collector_->hasResult("latency_test"));
}

TEST_F(ResultCollectorTest, test_get_results_nonexistent_name) {
    // Given: Empty collector

    // When: Query non-existent result
    auto results = collector_->getResults("nonexistent");

    // Then: Return empty vector
    EXPECT_TRUE(results.empty());
}

TEST_F(ResultCollectorTest, test_clear_removes_all_results) {
    // Given: Collector with results
    for (int i = 0; i < 3; ++i) {
        BenchmarkResult result;
        result.name = "test" + std::to_string(i);
        result.value = static_cast<float>(i);
        result.valid = true;
        collector_->addResult(result);
    }
    ASSERT_EQ(collector_->getResultCount(), 3);

    // When: Clear collector
    collector_->clear();

    // Then: All results removed
    EXPECT_EQ(collector_->getResultCount(), 0);
    EXPECT_FALSE(collector_->hasResult("test0"));
}

TEST_F(ResultCollectorTest, test_get_average_value) {
    // Given: Multiple results
    for (int i = 0; i < 5; ++i) {
        BenchmarkResult result;
        result.name = "avg_test";
        result.value = 100.0f + i * 10.0f;  // 100, 110, 120, 130, 140
        result.valid = true;
        collector_->addResult(result);
    }

    // When: Calculate average
    float average = collector_->getAverageValue("avg_test");

    // Then: Average is correct (120.0)
    EXPECT_FLOAT_EQ(average, 120.0f);
}

TEST_F(ResultCollectorTest, test_get_average_empty_collector) {
    // Given: Empty collector

    // When: Get average for non-existent result
    float average = collector_->getAverageValue("nonexistent");

    // Then: Return 0
    EXPECT_FLOAT_EQ(average, 0.0f);
}

TEST_F(ResultCollectorTest, test_result_with_statistics) {
    // Given: Result with statistical data
    BenchmarkResult result;
    result.name = "stat_test";
    result.value = 100.0f;
    result.min = 90.0f;
    result.max = 110.0f;
    result.mean = 100.0f;
    result.stddev = 5.0f;
    result.iterations = 100;
    result.valid = true;

    collector_->addResult(result);

    // Then: Statistics preserved
    auto results = collector_->getResults("stat_test");
    ASSERT_EQ(results.size(), 1);
    EXPECT_FLOAT_EQ(results[0].min, 90.0f);
    EXPECT_FLOAT_EQ(results[0].max, 110.0f);
    EXPECT_FLOAT_EQ(results[0].mean, 100.0f);
    EXPECT_FLOAT_EQ(results[0].stddev, 5.0f);
    EXPECT_EQ(results[0].iterations, 100);
}

}  // namespace cpm
