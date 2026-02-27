# PTX 快速参考卡片

## 内联PTX基础模板

```cuda
asm volatile (
    "指令模板"
    : 输出操作数列表
    : 输入操作数列表
    : clobber列表
);
```

## 约束字符速查

| 约束 | 类型 | 说明 |
|------|------|------|
| `r` | 32位 | 通用寄存器 (int, uint32_t, float) |
| `l` | 64位 | 通用寄存器 (int64_t, 指针) |
| `f` | 32位 | 浮点寄存器 (float) |
| `d` | 64位 | 浮点寄存器 (double) |
| `h` | 16位 | 浮点寄存器 (half, __half) |
| `c` | 1位 | 谓词寄存器 (bool) |
| `n` | 立即数 | 编译时常量 |
| `m` | 内存 | 内存操作数 |

## 常用PTX指令

### 内存访问指令

```cuda
// 加载指令
ld.global.f32    %dst, [%addr]      // 全局内存加载
ld.shared.f32    %dst, [%addr]      // 共享内存加载
ld.local.f32     %dst, [%addr]      // 本地内存加载
ld.const.f32     %dst, [%addr]      // 常量内存加载

// 存储指令
st.global.f32    [%addr], %src      // 全局内存存储
st.shared.f32    [%addr], %src      // 共享内存存储

// 缓存修饰符
.ca   // Cache All - 缓存到L1和L2
.cg   // Cache Global - 仅缓存到L2
.cs   // Cache Streaming - 流式访问
.lu   // Last Use - 提示数据不再使用
.cv   // Cache Volatile - 易失性访问

// 示例
asm volatile ("ld.global.ca.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
asm volatile ("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
```

### 算术指令

```cuda
// 整数运算
add.u32          %dst, %src1, %src2
sub.u32          %dst, %src1, %src2
mul.lo.u32       %dst, %src1, %src2  // 低32位
mul.hi.u32       %dst, %src1, %src2  // 高32位
mad.lo.u32       %dst, %src1, %src2, %src3  // 乘加

// 浮点运算
add.f32          %dst, %src1, %src2
sub.f32          %dst, %src1, %src2
mul.f32          %dst, %src1, %src2
fma.rn.f32       %dst, %src1, %src2, %src3  // 融合乘加
rcp.approx.f32   %dst, %src           // 近似倒数
sqrt.approx.f32  %dst, %src           // 近似平方根

// 舍入模式修饰符
.rn   // Round to nearest even
.rz   // Round toward zero
.rm   // Round toward negative infinity
.rp   // Round toward positive infinity
.approx // 近似计算
```

### 原子操作指令

```cuda
// 原子加
atom.global.add.u32   %old, [%addr], %val
atom.shared.add.u32   %old, [%addr], %val

// 原子比较交换
atom.global.cas.u32   %old, [%addr], %cmp, %val

// 原子交换
atom.global.exch.u32  %old, [%addr], %val

// 原子最大/最小
atom.global.max.u32   %old, [%addr], %val
atom.global.min.u32   %old, [%addr], %val
```

### 同步与屏障指令

```cuda
// 线程块内屏障
bar.sync          %barrier_id, %num_threads

// 内存屏障
membar.cta        // CTA级别内存屏障
membar.gl         // 全局内存屏障
membar.sys        // 系统级内存屏障

// 内存围栏
fence.sc.cta      // CTA级别顺序一致性围栏
fence.sc.gl       // 全局级别顺序一致性围栏
```

### 特殊寄存器访问

```cuda
// 读取特殊寄存器
mov.u32           %dst, %ctaid.x     // blockIdx.x
mov.u32           %dst, %ctaid.y     // blockIdx.y
mov.u32           %dst, %tid.x       // threadIdx.x
mov.u32           %dst, %tid.y       // threadIdx.y
mov.u32           %dst, %ntid.x      // blockDim.x
mov.u32           %dst, %nctaid.x    // gridDim.x
mov.u32           %dst, %clock       // 时钟计数器
mov.u32           %dst, %clock64     // 64位时钟计数器
mov.u32           %dst, %laneid      // warp内线程ID (0-31)
warpid            // warp ID
smid              // SM ID
```

### 谓词操作

```cuda
// 设置谓词
setp.eq.f32       %p, %src1, %src2   // 相等
setp.ne.f32       %p, %src1, %src2   // 不等
setp.lt.f32       %p, %src1, %src2   // 小于
setp.le.f32       %p, %src1, %src2   // 小于等于
setp.gt.f32       %p, %src1, %src2   // 大于
setp.ge.f32       %p, %src1, %src2   // 大于等于

// 条件执行
@%p               // 如果谓词为真执行
@!%p              // 如果谓词为假执行
```

## 完整示例

### 示例1: 精确计时器

```cuda
__device__ uint64_t get_clock() {
    uint64_t clock;
    asm volatile ("mov.u64 %0, %clock64;" : "=l"(clock));
    return clock;
}

__global__ void measure_latency(float* data, int* indices, int num_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint64_t start = get_clock();
    
    float sum = 0.0f;
    int curr = indices[idx];
    
    #pragma unroll
    for (int i = 0; i < num_iters; i++) {
        float val;
        asm volatile (
            "ld.global.ca.f32 %0, [%1];"
            : "=f"(val)
            : "l"(&data[curr])
        );
        sum += val;
        curr = __float2int_rn(val);
    }
    
    uint64_t end = get_clock();
    
    // 存储结果防止优化
    if (sum == 0.0f) {
        data[0] = (float)(end - start);
    }
}
```

### 示例2: 带缓存修饰符的内存访问

```cuda
__device__ float load_with_cache_policy(float* ptr, int policy) {
    float val;
    if (policy == 0) {
        // L1 + L2缓存
        asm volatile ("ld.global.ca.f32 %0, [%1];" 
                      : "=f"(val) : "l"(ptr));
    } else if (policy == 1) {
        // 仅L2缓存
        asm volatile ("ld.global.cg.f32 %0, [%1];" 
                      : "=f"(val) : "l"(ptr));
    } else {
        // 不缓存
        asm volatile ("ld.global.cs.f32 %0, [%1];" 
                      : "=f"(val) : "l"(ptr));
    }
    return val;
}
```

### 示例3: 原子操作包装

```cuda
__device__ uint32_t atomic_add_uint32(uint32_t* addr, uint32_t val) {
    uint32_t old;
    asm volatile (
        "atom.global.add.u32 %0, [%1], %2;"
        : "=r"(old)
        : "l"(addr), "r"(val)
        : "memory"
    );
    return old;
}

__device__ uint32_t atomic_cas_uint32(uint32_t* addr, 
                                       uint32_t cmp, 
                                       uint32_t val) {
    uint32_t old;
    asm volatile (
        "atom.global.cas.u32 %0, [%1], %2, %3;"
        : "=r"(old)
        : "l"(addr), "r"(cmp), "r"(val)
        : "memory"
    );
    return old;
}
```

### 示例4: 内存屏障使用

```cuda
__device__ void producer_consumer_sync() {
    __shared__ int flag;
    
    if (threadIdx.x == 0) {
        // 生产者写入数据
        // ... 写入共享内存 ...
        
        // 内存屏障确保写入完成
        asm volatile ("membar.cta;" ::: "memory");
        
        // 设置标志
        flag = 1;
    } else {
        // 消费者等待标志
        int local_flag;
        do {
            asm volatile ("ld.shared.u32 %0, [%1];"
                          : "=r"(local_flag)
                          : "l"(&flag));
        } while (local_flag == 0);
        
        // 内存屏障确保读取顺序
        asm volatile ("membar.cta;" ::: "memory");
        
        // ... 读取共享内存 ...
    }
}
```

## 常见陷阱

### 陷阱1: 忘记volatile

```cuda
// BAD: 可能被编译器优化掉
asm ("membar.gl;");

// GOOD: 防止优化
asm volatile ("membar.gl;");
```

### 陷阱2: 约束不匹配

```cuda
// BAD: 64位指针使用r约束
asm ("ld.global.f32 %0, [%1];" : "=f"(val) : "r"(ptr));

// GOOD: 64位指针使用l约束
asm ("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
```

### 陷阱3: 缺少memory clobber

```cuda
// BAD: 编译器可能重排内存操作
asm ("st.global.f32 [%0], %1;" :: "l"(ptr), "f"(val));

// GOOD: 声明内存修改
asm ("st.global.f32 [%0], %1;" :: "l"(ptr), "f"(val) : "memory");
```

### 陷阱4: 谓词寄存器冲突

```cuda
// BAD: 可能与其他谓词冲突
asm ("setp.gt.f32 %p1, %0, %1;" :: "f"(a), "f"(b));

// GOOD: 让编译器分配谓词
asm ("setp.gt.f32 %p, %0, %1;" :: "f"(a), "f"(b) : "p");
```
