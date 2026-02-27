# CUDA+PTX Microbenchmark TDD 工作流程

## 概述

本文档为 NVIDIA 4070 GPU 的 microbenchmark 工程提供完整的测试驱动开发（TDD）指导，涵盖 L1/L2 cache 和各种 PTX 指令的性能测试。

## TDD 核心原则

### Red-Green-Refactor 循环

1. **RED**: 编写失败的测试（描述期望行为）
2. **GREEN**: 编写最小实现使测试通过
3. **REFACTOR**: 优化代码，保持测试通过
4. **COVERAGE**: 验证覆盖率 80%+

## 测试策略

### 1. PTX 汇编代码正确性验证

PTX 代码测试采用分层策略：

| 层级 | 测试内容 | 方法 |
|------|---------|------|
| 语法层 | PTX 指令格式正确性 | `ptxas` 汇编验证 |
| 语义层 | 指令行为符合预期 | 运行验证 kernel |
| 性能层 | 执行周期/吞吐量 | 重复执行测量 |

**示例测试场景**：
- `LDG` 指令：验证全局内存加载正确性
- `LDS` 指令：验证共享内存加载正确性
- `FMA` 指令：验证乘加运算精度
- `BAR.SYNC` 指令：验证线程同步行为

### 2. 性能测量代码准确性验证

性能测试的关键挑战：

```cpp
// 需要验证的测量代码模式
class GpuTimer {
public:
    void start();  // 需要测试：是否记录正确时间戳
    void stop();   // 需要测试：是否计算正确耗时
    float elapsed_ms() const;  // 需要测试：精度是否足够
};
```

**验证方法**：
1. **已知时间验证**：使用 `__nanosleep()` 测试计时器精度
2. **统计验证**：多次测量验证方差在合理范围
3. **交叉验证**：与 `nvprof`/`ncu` 结果对比

### 3. 输出结果合理性验证

结果验证检查点：

- **数值范围**：带宽是否在理论峰值范围内（4070: 504 GB/s）
- **一致性**：多次运行结果方差 < 5%
- **单调性**：数据量增大时，带宽不应异常下降
- **边界条件**：空数据、单元素、对齐/非对齐访问

## 测试框架选择

### 推荐：Google Test + Google Mock

**理由**：
- CUDA 社区广泛使用
- 支持参数化测试（适合不同数据大小）
- 死亡测试支持（验证错误处理）
- 与 CMake 集成良好

**替代方案对比**：

| 框架 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| Google Test | 功能全面，社区大 | 编译时间较长 | 首选方案 |
| Catch2 | 头文件-only，轻量 | CUDA 支持较弱 | 简单项目 |
| doctest | 编译速度快 | 功能较少 | 快速原型 |

## 测试目录结构

```
cuda-ptx-microbenchmark/
├── src/                          # 源代码
│   ├── kernels/                  # CUDA kernels
│   │   ├── memory/               # 内存操作 kernels
│   │   │   ├── l1_cache.cu       # L1 cache benchmark
│   │   │   ├── l2_cache.cu       # L2 cache benchmark
│   │   │   └── global_memory.cu  # 全局内存 benchmark
│   │   ├── ptx/                  # PTX 指令 kernels
│   │   │   ├── arithmetic.cu     # 算术指令
│   │   │   ├── memory_ptx.cu     # 内存指令
│   │   │   └── synchronization.cu # 同步指令
│   │   └── utils/                # 工具 kernels
│   ├── core/                     # 核心功能
│   │   ├── timer.cpp             # GPU 计时器
│   │   ├── benchmark_runner.cpp  # 基准测试运行器
│   │   └── result_collector.cpp  # 结果收集器
│   └── main.cpp                  # 主程序入口
│
├── include/                      # 头文件
│   ├── kernels/                  # Kernel 接口
│   ├── core/                     # 核心接口
│   └── ptx/                      # PTX 内联汇编接口
│
├── tests/                        # 测试代码
│   ├── unit/                     # 单元测试
│   │   ├── test_timer.cpp        # 计时器测试
│   │   ├── test_result_collector.cpp
│   │   └── test_ptx_assembler.cpp
│   ├── integration/              # 集成测试
│   │   ├── test_l1_cache.cpp     # L1 cache 完整测试
│   │   ├── test_l2_cache.cpp     # L2 cache 完整测试
│   │   ├── test_ptx_instructions.cpp
│   │   └── test_memory_bandwidth.cpp
│   ├── e2e/                      # 端到端测试
│   │   ├── test_full_benchmark_suite.cpp
│   │   └── test_report_generation.cpp
│   └── fixtures/                 # 测试夹具
│       ├── gpu_test_fixture.h    # GPU 测试基类
│       └── benchmark_fixture.h   # Benchmark 测试基类
│
├── cmake/                        # CMake 模块
│   ├── FindCUDA.cmake
│   └── CodeCoverage.cmake
│
└── scripts/                      # 辅助脚本
    ├── run_tests.sh              # 测试运行脚本
    ├── check_coverage.sh         # 覆盖率检查脚本
    └── verify_ptx.sh             # PTX 验证脚本
```

## 命名规范

### 文件命名

| 类型 | 命名模式 | 示例 |
|------|---------|------|
| 源文件 | `snake_case.cu/cpp` | `l1_cache.cu` |
| 头文件 | `snake_case.h` | `gpu_timer.h` |
| 测试文件 | `test_*.cpp` | `test_timer.cpp` |
| PTX 文件 | `*.ptx` | `fma_latency.ptx` |

### 测试命名

```cpp
// 测试类命名：被测对象 + Test
class GpuTimerTest : public ::testing::Test {};

// 测试函数命名：test_ + 场景_ + 期望结果
TEST_F(GpuTimerTest, test_start_stop_returns_positive_elapsed_time);
TEST_F(GpuTimerTest, test_multiple_iterations_accumulate_correctly);
TEST_F(GpuTimerTest, test_nanosleep_measures_accurate_duration);

// PTX 测试命名
TEST_P(PtxInstructionTest, test_fma_instruction_produces_correct_result);
TEST_P(PtxInstructionTest, test_ldg_instruction_loads_global_memory);
```

## 测试文件模板

### 1. 单元测试模板

```cpp
// tests/unit/test_timer.cpp
#include <gtest/gtest.h>
#include "core/gpu_timer.h"
#include "tests/fixtures/gpu_test_fixture.h"

using namespace cpm;

class GpuTimerTest : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();
        timer_ = std::make_unique<GpuTimer>();
    }

    void TearDown() override {
        timer_.reset();
        GpuTestFixture::TearDown();
    }

    std::unique_ptr<GpuTimer> timer_;
};

// RED: 首先编写失败测试
TEST_F(GpuTimerTest, test_start_stop_returns_positive_elapsed_time) {
    // Given: 计时器已创建
    ASSERT_NE(timer_, nullptr);

    // When: 启动并立即停止
    timer_->start();
    timer_->stop();

    // Then: 经过时间应为正值
    float elapsed = timer_->elapsed_ms();
    EXPECT_GT(elapsed, 0.0f);
    EXPECT_LT(elapsed, 1.0f);  // 应该小于1ms
}

TEST_F(GpuTimerTest, test_nanosleep_measures_accurate_duration) {
    // Given: 已知休眠时间
    const int sleep_ns = 1000000;  // 1ms
    const float tolerance = 0.1f;   // 10% 容差

    // When: 测量休眠时间
    timer_->start();
    nanosleep_kernel<<<1, 1>>>(sleep_ns);
    cudaDeviceSynchronize();
    timer_->stop();

    // Then: 测量值应在预期范围内
    float elapsed_ms = timer_->elapsed_ms();
    float expected_ms = sleep_ns / 1e6f;
    EXPECT_NEAR(elapsed_ms, expected_ms, expected_ms * tolerance);
}

// 参数化测试：不同迭代次数
class GpuTimerIterationsTest : public GpuTimerTest,
                               public ::testing::WithParamInterface<int> {};

TEST_P(GpuTimerIterationsTest, test_multiple_iterations_accumulate_correctly) {
    int iterations = GetParam();

    timer_->start();
    for (int i = 0; i < iterations; ++i) {
        dummy_kernel<<<1, 1>>>();
    }
    cudaDeviceSynchronize();
    timer_->stop();

    float total_time = timer_->elapsed_ms();
    EXPECT_GT(total_time, 0.0f);

    // 更多迭代应该花费更多时间（粗略检查）
    if (iterations > 1) {
        timer_->start();
        dummy_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        timer_->stop();
        float single_time = timer_->elapsed_ms();
        EXPECT_GT(total_time, single_time);
    }
}

INSTANTIATE_TEST_SUITE_P(
    IterationCounts,
    GpuTimerIterationsTest,
    ::testing::Values(1, 10, 100, 1000)
);
```

### 2. PTX 指令测试模板

```cpp
// tests/integration/test_ptx_instructions.cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "kernels/ptx/arithmetic.h"
#include "tests/fixtures/gpu_test_fixture.h"

using namespace cpm;

// FMA 指令测试
class PtxFmaTest : public GpuTestFixture {};

TEST_F(PtxFmaTest, test_fma_single_precision_computes_correctly) {
    // Given: 输入数据
    float a = 2.0f, b = 3.0f, c = 4.0f;
    float expected = a * b + c;  // 10.0f
    float result = 0.0f;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // When: 执行 PTX FMA 指令
    fma_kernel<<<1, 1>>>(a, b, c, d_result);
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    // Then: 结果正确
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(PtxFmaTest, test_fma_latency_measurement_is_reasonable) {
    // Given: 测量参数
    const int iterations = 10000;
    const float expected_latency_ns = 4.0f;  // FMA 典型延迟
    const float tolerance = 0.5f;             // 50% 容差（考虑调度）

    // When: 测量 FMA 延迟
    float measured_latency = measure_fma_latency(iterations);

    // Then: 测量值在合理范围
    EXPECT_GT(measured_latency, 0.0f);
    EXPECT_LT(measured_latency, expected_latency_ns * (1 + tolerance));
}

// 内存指令测试
class PtxMemoryTest : public GpuTestFixture {
protected:
    float* d_data_ = nullptr;
    const size_t kDataSize = 1024;

    void SetUp() override {
        GpuTestFixture::SetUp();
        cudaMalloc(&d_data_, kDataSize * sizeof(float));
        std::vector<float> host_data(kDataSize, 1.0f);
        cudaMemcpy(d_data_, host_data.data(), kDataSize * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_data_);
        GpuTestFixture::TearDown();
    }
};

TEST_F(PtxMemoryTest, test_ldg_loads_global_memory_correctly) {
    // Given: 已初始化的设备内存
    float result = 0.0f;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // When: 使用 LDG 加载
    ldg_kernel<<<1, 1>>>(d_data_, d_result);
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    // Then: 加载值正确
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST_F(PtxMemoryTest, test_lds_loads_shared_memory_correctly) {
    // Given: 共享内存已初始化
    const int threads = 32;
    float result = 0.0f;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // When: 使用 LDS 加载共享内存
    lds_kernel<<<1, threads>>>(d_result);
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    // Then: 结果正确
    EXPECT_FLOAT_EQ(result, 42.0f);  // kernel 中设置的值
}
```

### 3. Cache Benchmark 测试模板

```cpp
// tests/integration/test_l1_cache.cpp
#include <gtest/gtest.h>
#include "kernels/memory/l1_cache.h"
#include "core/benchmark_runner.h"
#include "tests/fixtures/benchmark_fixture.h"

using namespace cpm;

class L1CacheBenchmarkTest : public BenchmarkTestFixture {
protected:
    static constexpr size_t kL1Size = 128 * 1024;  // 128KB L1 cache
    static constexpr size_t kCacheLine = 128;       // 128B cache line
};

TEST_F(L1CacheBenchmarkTest, test_l1_hit_rate_with_sequential_access) {
    // Given: 适合 L1 的数据大小
    const size_t data_size = kL1Size / 2;
    L1CacheBenchmark benchmark(data_size);

    // When: 执行顺序访问测试
    auto result = benchmark.run_sequential_access();

    // Then: 命中率应接近 100%
    EXPECT_GT(result.hit_rate, 0.95f);
    EXPECT_GT(result.bandwidth_gbps, 1000.0f);  // L1 带宽应很高
}

TEST_F(L1CacheBenchmarkTest, test_l1_miss_with_random_access) {
    // Given: 超过 L1 大小的数据
    const size_t data_size = kL1Size * 4;
    L1CacheBenchmark benchmark(data_size);

    // When: 执行随机访问测试
    auto result = benchmark.run_random_access();

    // Then: 应有显著数量的 miss
    EXPECT_LT(result.hit_rate, 0.5f);
    EXPECT_LT(result.bandwidth_gbps, 500.0f);  // 带宽应降低
}

TEST_F(L1CacheBenchmarkTest, test_bandwidth_within_theoretical_limits) {
    // Given: 4070 理论 L1 带宽约 10+ TB/s
    const float theoretical_max = 12000.0f;  // GB/s
    L1CacheBenchmark benchmark(kL1Size / 2);

    // When: 测量带宽
    auto result = benchmark.run_sequential_access();

    // Then: 测量值不应超过理论峰值
    EXPECT_LT(result.bandwidth_gbps, theoretical_max * 1.1f);
    EXPECT_GT(result.bandwidth_gbps, 0.0f);
}

// 边界条件测试
TEST_F(L1CacheBenchmarkTest, test_empty_data_handles_gracefully) {
    L1CacheBenchmark benchmark(0);
    auto result = benchmark.run_sequential_access();

    EXPECT_EQ(result.status, BenchmarkStatus::kSkipped);
}

TEST_F(L1CacheBenchmarkTest, test_unaligned_access_works_correctly) {
    // 非对齐访问测试
    L1CacheBenchmark benchmark(kL1Size, /* alignment */ 1);
    auto result = benchmark.run_sequential_access();

    EXPECT_EQ(result.status, BenchmarkStatus::kSuccess);
}
```

### 4. 测试夹具基类

```cpp
// tests/fixtures/gpu_test_fixture.h
#pragma once

#include <gtest/gtest.h>
#include <cuda_runtime.h>

namespace cpm {

class GpuTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // 检查 CUDA 设备可用
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        ASSERT_EQ(err, cudaSuccess) << "Failed to get CUDA device count";
        ASSERT_GT(device_count, 0) << "No CUDA devices found";

        // 获取当前设备属性
        cudaGetDevice(&device_id_);
        cudaGetDeviceProperties(&device_props_, device_id_);

        // 重置设备状态
        cudaDeviceReset();
    }

    void TearDown() override {
        // 检查是否有 CUDA 错误残留
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error at teardown: " << cudaGetErrorString(err) << std::endl;
        }

        // 同步确保所有操作完成
        cudaDeviceSynchronize();
    }

    int device_id_ = 0;
    cudaDeviceProp device_props_;
};

// 检查特定计算能力
#define SKIP_IF_COMPUTE_LESS_THAN(major, minor) \
    do { \
        if (device_props_.major < major || \
            (device_props_.major == major && device_props_.minor < minor)) { \
            GTEST_SKIP() << "Requires compute capability " << major << "." << minor; \
        } \
    } while(0)

}  // namespace cpm
```

```cpp
// tests/fixtures/benchmark_fixture.h
#pragma once

#include "gpu_test_fixture.h"
#include <vector>

namespace cpm {

class BenchmarkTestFixture : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();

        // 检查是否为 4070 或兼容设备
        if (device_props_.major != 8 || device_props_.minor != 9) {
            std::cout << "Warning: Running on " << device_props_.name
                      << " (compute " << device_props_.major << "."
                      << device_props_.minor << "), expected Ada (8.9)" << std::endl;
        }

        // 设置确定性模式
        cudaDeviceSetLimit(cudaLimitStackSize, 8192);
    }

    // 验证结果一致性
    template<typename T>
    bool results_are_consistent(const std::vector<T>& results, float tolerance) {
        if (results.size() < 2) return true;

        T mean = std::accumulate(results.begin(), results.end(), T(0));
        mean /= results.size();

        for (const auto& r : results) {
            if (std::abs(r - mean) > mean * tolerance) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace cpm
```

## CMake 配置

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(CudaPtxMicrobenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)  # Ada Lovelace (4070)

# 启用测试
enable_testing()

# 查找包
find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)

# 代码覆盖率（CPU 部分）
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler --coverage")
endif()

# 主库
add_library(cpm_core
    src/core/timer.cpp
    src/core/benchmark_runner.cpp
    src/core/result_collector.cpp
)
target_include_directories(cpm_core PUBLIC include)

# CUDA kernels
add_library(cpm_kernels
    src/kernels/memory/l1_cache.cu
    src/kernels/memory/l2_cache.cu
    src/kernels/memory/global_memory.cu
    src/kernels/ptx/arithmetic.cu
    src/kernels/ptx/memory_ptx.cu
)
target_link_libraries(cpm_kernels cpm_core)

# 测试
add_subdirectory(tests)
```

```cmake
# tests/CMakeLists.txt

# 测试工具库
add_library(test_fixtures
    fixtures/gpu_test_fixture.cpp
    fixtures/benchmark_fixture.cpp
)
target_link_libraries(test_fixtures GTest::gtest cpm_core)

# 单元测试
add_executable(unit_tests
    unit/test_timer.cpp
    unit/test_result_collector.cpp
    unit/test_ptx_assembler.cpp
)
target_link_libraries(unit_tests test_fixtures cpm_kernels)
add_test(NAME UnitTests COMMAND unit_tests)

# 集成测试
add_executable(integration_tests
    integration/test_l1_cache.cpp
    integration/test_l2_cache.cpp
    integration/test_ptx_instructions.cpp
    integration/test_memory_bandwidth.cpp
)
target_link_libraries(integration_tests test_fixtures cpm_kernels)
add_test(NAME IntegrationTests COMMAND integration_tests)

# E2E 测试
add_executable(e2e_tests
    e2e/test_full_benchmark_suite.cpp
    e2e/test_report_generation.cpp
)
target_link_libraries(e2e_tests test_fixtures cpm_kernels)
add_test(NAME E2ETests COMMAND e2e_tests)
```

## 测试运行命令

### 基本测试运行

```bash
# 构建项目
mkdir build && cd build
cmake .. -DENABLE_TESTING=ON
make -j$(nproc)

# 运行所有测试
ctest --output-on-failure

# 运行特定测试套件
./tests/unit_tests
./tests/integration_tests --gtest_filter="*L1Cache*"

# 详细输出
./tests/unit_tests --gtest_also_run_disabled_tests --gtest_repeat=3
```

### 覆盖率检查

```bash
# 配置时启用覆盖率
cmake .. -DENABLE_COVERAGE=ON
make
ctest

# 生成覆盖率报告
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/tests/*' --output-file coverage_filtered.info
genhtml coverage_filtered.info --output-directory coverage_report

# 检查覆盖率阈值
lcov --summary coverage_filtered.info | grep "lines.*:"
```

### PTX 验证脚本

```bash
#!/bin/bash
# scripts/verify_ptx.sh

set -e

PTX_DIR="src/kernels/ptx"

for ptx_file in $PTX_DIR/*.ptx; do
    echo "Verifying: $ptx_file"

    # 使用 ptxas 验证语法
    ptxas -arch=sm_89 "$ptx_file" -o /dev/null 2>&1 || {
        echo "ERROR: PTX assembly failed for $ptx_file"
        exit 1
    }

done

echo "All PTX files verified successfully"
```

## 常见陷阱与注意事项

### 1. CUDA 上下文管理

```cpp
// WRONG: 测试间共享上下文可能导致状态泄漏
TEST(SomeTest, First) {
    cudaMalloc(&ptr, size);  // 未释放
}

// CORRECT: 每个测试独立清理
TEST(SomeTest, First) {
    void* ptr;
    cudaMalloc(&ptr, size);
    // ... 测试逻辑
    cudaFree(ptr);  // 确保释放
}
```

### 2. 异步操作同步

```cpp
// WRONG: 未同步就检查错误
kernel<<<grid, block>>>(data);
EXPECT_EQ(cudaGetLastError(), cudaSuccess);  // 可能检查太早

// CORRECT: 同步后再检查
cudaDeviceSynchronize();
EXPECT_EQ(cudaGetLastError(), cudaSuccess);
```

### 3. 浮点精度比较

```cpp
// WRONG: 直接比较浮点数
EXPECT_EQ(gpu_result, 1.0f);

// CORRECT: 使用近似比较
EXPECT_FLOAT_EQ(gpu_result, 1.0f);        // 4 ULP 容差
EXPECT_NEAR(gpu_result, 1.0f, 1e-5);      // 绝对容差
```

### 4. 设备内存管理

```cpp
// 使用 RAII 包装避免泄漏
template<typename T>
class DeviceBuffer {
    T* ptr_ = nullptr;
    size_t size_ = 0;
public:
    explicit DeviceBuffer(size_t count) : size_(count) {
        cudaMalloc(&ptr_, count * sizeof(T));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }

    // 禁止拷贝，允许移动
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
    }

    T* get() const { return ptr_; }
    size_t size() const { return size_; }
};

// 测试中使用
TEST_F(MyTest, test_with_buffer) {
    DeviceBuffer<float> buffer(1024);
    kernel<<<1, 1>>>(buffer.get());
    // 自动释放，不会泄漏
}
```

### 5. 性能测试的统计方法

```cpp
// WRONG: 单次测量
TEST(PerfTest, Unreliable) {
    timer.start();
    kernel<<<1,1>>>();
    timer.stop();
    EXPECT_LT(timer.elapsed(), expected);  // 可能因调度波动失败
}

// CORRECT: 多次测量取中位数
TEST(PerfTest, Reliable) {
    std::vector<float> times;
    for (int i = 0; i < 100; ++i) {
        timer.start();
        kernel<<<1,1>>>();
        timer.stop();
        times.push_back(timer.elapsed());
    }

    std::sort(times.begin(), times.end());
    float median = times[times.size() / 2];

    // 使用中位数而非平均值，对异常值更鲁棒
    EXPECT_LT(median, expected * 1.1f);
}
```

### 6. PTX 内联汇编测试

```cpp
// 测试 PTX 指令时，验证行为和性能
TEST(PtxTest, TestInstructionBehaviorAndPerformance) {
    // 1. 验证正确性
    float result;
    ptx_kernel<<<1,1>>>(input, &result);
    cudaDeviceSynchronize();
    EXPECT_FLOAT_EQ(result, expected);

    // 2. 验证性能特征
    auto perf = measure_instruction_throughput();
    EXPECT_GT(perf.throughput, theoretical_max * 0.8f);
    EXPECT_LT(perf.latency, expected_latency * 1.2f);
}
```

## 覆盖率目标达成策略

### 测量难点

CUDA kernel 代码的覆盖率测量有挑战：
- `nvcc` 生成的代码包含大量模板展开
- 设备代码执行路径难以追踪

### 应对策略

1. **分离逻辑**：将可测试逻辑移到主机代码
   ```cpp
   // 可测试的主机函数
   __host__ int calculate_grid_size(int data_size, int block_size) {
       return (data_size + block_size - 1) / block_size;
   }

   // 简单的 kernel 包装
   __global__ void kernel_wrapper(float* data, int size) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < size) {
           actual_kernel(data[idx]);
       }
   }
   ```

2. **使用 `gcov` + `nvcc -Xcompiler --coverage`** 对主机代码

3. **使用 `cuobjdump` 验证设备代码路径**

4. **手动验证关键路径**：对关键 kernel，编写特定测试覆盖每个分支

### 覆盖率检查清单

- [ ] 所有公共函数有单元测试
- [ ] 所有 kernel 有集成测试
- [ ] 所有 PTX 指令有行为和性能测试
- [ ] 所有错误处理路径有测试
- [ ] 边界条件（空数据、最大值、非对齐）有测试
- [ ] 主机代码覆盖率 80%+
- [ ] 关键设备代码路径手动验证

## 总结

遵循本文档的 TDD 工作流程：

1. **先写测试**：每个功能前编写失败测试
2. **分层测试**：单元 + 集成 + E2E
3. **验证行为**：PTX 指令正确性和性能特征
4. **边界覆盖**：空数据、非对齐、大负载
5. **统计鲁棒**：多次测量，中位数而非平均值
6. **持续集成**：自动化测试和覆盖率检查

这将确保 CUDA+PTX microbenchmark 代码的正确性、可靠性和可维护性。
