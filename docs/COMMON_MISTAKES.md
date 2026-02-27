# CUDA + PTX 常见错误与解决方案

## 1. CUDA错误处理类

### 错误1.1: 忽略异步错误

**错误代码：**
```cuda
// BAD: 只检查同步错误
cudaMalloc(&d_data, size);
kernel<<<blocks, threads>>>(d_data);
cudaError_t err = cudaGetLastError();  // 只能捕获启动错误
if (err != cudaSuccess) { /* 处理错误 */ }
// 内核执行错误被忽略！
```

**正确代码：**
```cuda
// GOOD: 检查同步和异步错误
CUDA_CHECK(cudaMalloc(&d_data, size));
kernel<<<blocks, threads>>>(d_data);
CUDA_CHECK(cudaGetLastError());      // 检查启动错误
CUDA_CHECK(cudaDeviceSynchronize()); // 检查执行错误
```

### 错误1.2: 内存分配失败未处理

**错误代码：**
```cuda
// BAD: 未检查分配结果
cudaMalloc(&d_data, huge_size);  // 可能失败
// 直接使用d_data，可能导致崩溃
kernel<<<blocks, threads>>>(d_data);
```

**正确代码：**
```cuda
// GOOD: 检查分配结果
cudaError_t err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate %zu bytes: %s\n", 
            size, cudaGetErrorString(err));
    // 回退策略：使用较小的尺寸或退出
    return -1;
}
```

### 错误1.3: 内存泄漏

**错误代码：**
```cuda
// BAD: 异常路径上未释放内存
void process() {
    float *d_data;
    cudaMalloc(&d_data, size);
    
    if (some_condition) {
        return;  // 内存泄漏！
    }
    
    cudaFree(d_data);
}
```

**正确代码：**
```cuda
// GOOD: 使用RAII确保释放
class CudaBuffer {
    float* ptr_;
public:
    explicit CudaBuffer(size_t size) {
        CUDA_CHECK(cudaMalloc(&ptr_, size));
    }
    ~CudaBuffer() { cudaFree(ptr_); }
    float* get() const { return ptr_; }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& o) : ptr_(o.ptr_) { o.ptr_ = nullptr; }
};

void process() {
    CudaBuffer buffer(size);  // 自动管理
    if (some_condition) {
        return;  // 自动释放
    }
}  // 自动释放
```

## 2. 内核设计类

### 错误2.1: 缺少边界检查

**错误代码：**
```cuda
// BAD: 无边界检查
__global__ void scale(float* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= factor;  // 越界风险！
}
```

**正确代码：**
```cuda
// GOOD: 边界检查
__global__ void scale(float* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;  // 边界检查
    data[idx] *= factor;
}
```

### 错误2.2: 线程发散

**错误代码：**
```cuda
// BAD: warp内线程走不同分支
__global__ void process(float* data, int n) {
    int idx = threadIdx.x;
    if (idx % 2 == 0) {
        data[idx] = sqrtf(data[idx]);
    } else {
        data[idx] = data[idx] * data[idx];
    }
}
```

**正确代码：**
```cuda
// GOOD: 按warp对齐分支
__global__ void process(float* data, int n) {
    int idx = threadIdx.x;
    int warp_id = idx / 32;
    
    if ((warp_id % 2) == 0) {
        // 整个warp走同一分支
        data[idx] = sqrtf(data[idx]);
    } else {
        data[idx] = data[idx] * data[idx];
    }
}
```

### 错误2.3: 不正确的内核启动配置

**错误代码：**
```cuda
// BAD: block大小不是warp大小的倍数
dim3 block(100);  // 100不是32的倍数
dim3 grid((n + 99) / 100);
kernel<<<grid, block>>>(data, n);
```

**正确代码：**
```cuda
// GOOD: block大小是32的倍数
const int BLOCK_SIZE = 128;  // 128 = 4 * 32
dim3 block(BLOCK_SIZE);
dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
kernel<<<grid, block>>>(data, n);
```

## 3. PTX汇编类

### 错误3.1: 约束字符错误

**错误代码：**
```cuda
// BAD: 64位指针使用r约束
uint64_t* ptr = ...;
uint64_t val;
asm volatile ("ld.global.u64 %0, [%1];" 
              : "=l"(val) 
              : "r"(ptr));  // 错误：64位指针需要l约束
```

**正确代码：**
```cuda
// GOOD: 64位指针使用l约束
uint64_t* ptr = ...;
uint64_t val;
asm volatile ("ld.global.u64 %0, [%1];" 
              : "=l"(val) 
              : "l"(ptr));  // 正确
```

### 错误3.2: 缺少memory clobber

**错误代码：**
```cuda
// BAD: 未声明内存修改
__device__ void store_value(float* ptr, float val) {
    asm volatile ("st.global.f32 [%0], %1;"
                  :: "l"(ptr), "f"(val));
    // 编译器可能优化掉后续读取
}
```

**正确代码：**
```cuda
// GOOD: 声明memory clobber
__device__ void store_value(float* ptr, float val) {
    asm volatile ("st.global.f32 [%0], %1;"
                  :: "l"(ptr), "f"(val)
                  : "memory");  // 声明内存修改
}
```

### 错误3.3: 输入输出寄存器冲突

**错误代码：**
```cuda
// BAD: 输出覆盖输入
int result, a = 5;
asm volatile (
    "add.u32 %0, %0, 1;"  // 危险：如果result和a同一寄存器
    : "=r"(result)
    : "r"(a)
);
```

**正确代码：**
```cuda
// GOOD: 使用早期clobber
int result, a = 5;
asm volatile (
    "add.u32 %0, %1, 1;"  // 明确使用不同寄存器
    : "=r"(result)
    : "r"(a)
);

// 或者使用早期clobber
asm volatile (
    "add.u32 %0, %0, 1;"
    : "=&r"(result)  // 早期clobber
    : "0"(a)         // 与输出0同一位置
);
```

### 错误3.4: 谓词寄存器未声明

**错误代码：**
```cuda
// BAD: 修改谓词寄存器未声明
float result, a, b;
asm volatile (
    "setp.gt.f32 %p1, %1, 0.0;\n\t"
    "@%p1 add.f32 %0, %2, 1.0;\n\t"
    "@!%p1 mov.f32 %0, %2;"
    : "=f"(result)
    : "f"(a), "f"(b)
    // 缺少p1声明
);
```

**正确代码：**
```cuda
// GOOD: 声明谓词寄存器修改
float result, a, b;
asm volatile (
    "setp.gt.f32 %p1, %1, 0.0;\n\t"
    "@%p1 add.f32 %0, %2, 1.0;\n\t"
    "@!%p1 mov.f32 %0, %2;"
    : "=f"(result)
    : "f"(a), "f"(b)
    : "p1"  // 声明谓词寄存器修改
);
```

### 错误3.5: PTX语法错误

**错误代码：**
```cuda
// BAD: 指令格式错误
float val;
asm volatile ("ld.global.f32 [%0], %1;"  // 错误：操作数顺序
              : "=f"(val)
              : "l"(ptr));
```

**正确代码：**
```cuda
// GOOD: 正确的指令格式
float val;
asm volatile ("ld.global.f32 %0, [%1];"  // 先目标后源
              : "=f"(val)
              : "l"(ptr));
```

## 4. 性能测试类

### 错误4.1: 未预热GPU

**错误代码：**
```cuda
// BAD: 直接测量，未考虑启动开销和频率调整
GpuTimer timer;
timer.start();
kernel<<<blocks, threads>>>(d_data);
timer.stop();
float time = timer.elapsed();  // 结果不稳定
```

**正确代码：**
```cuda
// GOOD: 充分预热
// 1. 预热运行
for (int i = 0; i < 10; i++) {
    kernel<<<blocks, threads>>>(d_data);
}
CUDA_CHECK(cudaDeviceSynchronize());

// 2. 等待频率稳定
usleep(100000);

// 3. 正式测量
GpuTimer timer;
timer.start();
for (int i = 0; i < 100; i++) {
    kernel<<<blocks, threads>>>(d_data);
}
timer.stop();
float avg_time = timer.elapsed() / 100.0f;
```

### 错误4.2: 编译器优化干扰

**错误代码：**
```cuda
// BAD: 测试循环可能被优化掉
__global__ void latency_test(float* data, int iterations) {
    float* ptr = data;
    for (int i = 0; i < iterations; i++) {
        ptr = (float*)(*ptr);  // 编译器可能优化
    }
}
```

**正确代码：**
```cuda
// GOOD: 防止编译器优化
__device__ __noinline__ float force_read(float* ptr) {
    volatile float val = *ptr;
    return val;
}

__global__ void latency_test(float* data, int iterations) {
    float* ptr = data;
    float sum = 0.0f;
    
    #pragma unroll 1  // 禁止展开
    for (int i = 0; i < iterations; i++) {
        sum += force_read(ptr);
        ptr = (float*)(*(unsigned long long*)ptr);
    }
    
    // 使用结果防止优化
    if (sum == 0.0f) {
        *(volatile float*)data = sum;
    }
}
```

### 错误4.3: 计时精度不足

**错误代码：**
```cuda
// BAD: 使用CPU计时器测量GPU操作
auto start = std::chrono::high_resolution_clock::now();
kernel<<<blocks, threads>>>(d_data);
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();
// 包含内核启动开销，精度不足
```

**正确代码：**
```cuda
// GOOD: 使用CUDA事件计时
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);
kernel<<<blocks, threads>>>(d_data);
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

float elapsed_ms;
cudaEventElapsedTime(&elapsed_ms, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### 错误4.4: 未消除干扰因素

**错误代码：**
```cuda
// BAD: 未考虑缓存状态影响
void benchmark_memory() {
    float* d_data;
    cudaMalloc(&d_data, size);
    
    // 直接测量，缓存状态不确定
    timer_start();
    read_kernel(d_data);
    timer_stop();
}
```

**正确代码：**
```cuda
// GOOD: 控制缓存状态
void benchmark_memory() {
    float* d_data;
    cudaMalloc(&d_data, size);
    
    // 1. 冷缓存测量 - 先清空缓存
    cudaMemset(d_data, 0, size);  // 可能清空部分缓存
    // 或使用其他方法使缓存失效
    
    timer_start();
    read_kernel(d_data);  // 冷缓存
    timer_stop();
    
    // 2. 热缓存测量 - 预热后测量
    read_kernel(d_data);  // 预热
    
    timer_start();
    read_kernel(d_data);  // 热缓存
    timer_stop();
}
```

## 5. 数值安全类

### 错误5.1: 除零风险

**错误代码：**
```cuda
// BAD: 未检查除数
__global__ void normalize(float* data, float* divisor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] /= divisor[idx];  // 可能除零
}
```

**正确代码：**
```cuda
// GOOD: 除零保护
__device__ float safe_divide(float a, float b) {
    const float EPSILON = 1e-7f;
    if (fabsf(b) < EPSILON) {
        return (a >= 0.0f) ? FLT_MAX : -FLT_MAX;
    }
    return a / b;
}

__global__ void normalize(float* data, float* divisor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = safe_divide(data[idx], divisor[idx]);
}
```

### 错误5.2: 数组越界

**错误代码：**
```cuda
// BAD: 未验证索引计算
__global__ void access_2d(float* matrix, int x, int y, int pitch) {
    int idx = y * pitch + x;  // 可能溢出
    matrix[idx] = 0.0f;
}
```

**正确代码：**
```cuda
// GOOD: 索引验证
__global__ void access_2d(float* matrix, int x, int y, 
                          int width, int height, int pitch) {
    // 验证坐标
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    
    // 安全计算索引
    size_t idx = (size_t)y * pitch + x;
    size_t max_idx = (size_t)height * pitch;
    
    if (idx >= max_idx) return;  // 额外保护
    
    matrix[idx] = 0.0f;
}
```

### 错误5.3: 整数溢出

**错误代码：**
```cuda
// BAD: 整数溢出风险
__global__ void large_array_access(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int large_idx = idx * 1000000;  // 可能溢出
    data[large_idx] = 0.0f;
}
```

**正确代码：**
```cuda
// GOOD: 使用64位索引
__global__ void large_array_access(float* data, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t large_idx = idx * 1000000ULL;
    
    if (large_idx >= n) return;
    data[large_idx] = 0.0f;
}
```

## 6. 4070特定优化类

### 错误6.1: 未针对SM 8.9编译

**错误代码：**
```bash
# BAD: 使用旧架构编译
nvcc -arch=sm_70 benchmark.cu -o benchmark  # 未利用4070特性
```

**正确代码：**
```bash
# GOOD: 针对4070编译
nvcc -arch=sm_89 -code=sm_89 -O3 benchmark.cu -o benchmark
```

### 错误6.2: 忽略L2缓存大小

**错误代码：**
```cuda
// BAD: 未考虑48MB L2缓存
__global__ void cache_test(float* data, size_t size) {
    // 测试100MB数据，超出L2容量
    for (size_t i = 0; i < size / sizeof(float); i++) {
        data[i] = data[i] * 2.0f;
    }
}
```

**正确代码：**
```cuda
// GOOD: 针对48MB L2优化
const size_t L2_CACHE_SIZE = 48ULL * 1024 * 1024;  // 48MB

__global__ void cache_test(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // 分块处理以适配L2
    for (size_t base = 0; base < size / sizeof(float); base += L2_CACHE_SIZE / sizeof(float)) {
        size_t limit = min(base + L2_CACHE_SIZE / sizeof(float), 
                           size / sizeof(float));
        for (size_t i = base + idx; i < limit; i += stride) {
            data[i] = data[i] * 2.0f;
        }
    }
}
```

### 错误6.3: 共享内存配置不当

**错误代码：**
```cuda
// BAD: 未考虑共享内存限制
__global__ void shared_memory_kernel() {
    __shared__ float large_buffer[65536];  // 256KB，超过128KB限制
    // ...
}
```

**正确代码：**
```cuda
// GOOD: 检查共享内存限制
// RTX 4070每SM最大128KB共享内存
const int MAX_SHARED_PER_BLOCK = 48 * 1024;  // 保守使用48KB

__global__ void shared_memory_kernel() {
    __shared__ float buffer[MAX_SHARED_PER_BLOCK / sizeof(float)];
    // ...
}

// 或使用动态共享内存
__global__ void dynamic_shared_kernel() {
    extern __shared__ float dynamic_buffer[];
    // ...
}
// 启动时指定大小
// kernel<<<grid, block, shared_size>>>(args);
```
