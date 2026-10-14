#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <type_traits>

namespace cmb {

static constexpr int NACC = 4;

// Generic throughput kernel: each thread maintains N independent accumulators
// and applies the instruction functor once per accumulator per iteration.
template <int N, typename T, typename F>
__global__ void kernel_inst_thp(F f, T init, int iters, T *out) {
  T acc[N];
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    acc[i] = init + static_cast<T>(threadIdx.x + i);
    if constexpr (std::is_same<T, unsigned long long>::value)
      acc[i] += static_cast<unsigned long long>((threadIdx.x + i) * 3 + 7) << 32;
  }

  #pragma unroll 1
  for (int r = 0; r < iters; ++r) {
    #pragma unroll
    for (int i = 0; i < N; ++i)
      f(acc[i], r);
  }

  // keep live
  T sink = T(0);
  #pragma unroll
  for (int i = 0; i < N; ++i)
    sink += acc[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = sink;
}

template <int N, typename T, typename F>
static double measure_inst_thp(F f, T init, const char *label,
                               const char *key) {
  DeviceInfo info = queryDevice(0);
  int threads = 256; // 8 warps per block
  int blocks = info.sm_count * 8;
  int iters = 1 << 12;
  T *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(T)));

  kernel_inst_thp<N><<<blocks, threads>>>(f, init, iters, d_out);
  CUDA_CHECK(cudaGetLastError());

  GpuTimer timer;
  timer.start();
  kernel_inst_thp<N><<<blocks, threads>>>(f, init, iters, d_out);
  timer.stop();
  CUDA_CHECK(cudaGetLastError());
  float ms = timer.elapsedMs();

  long long total_inst = (long long)blocks * threads * N * iters;
  double hz = (double)info.clock_rate_khz * 1000.0;
  double ipc = (total_inst / 32.0 / (double)info.sm_count) / ((ms * 1e-3) * hz);
  printf("  %s throughput: %.2f inst/cycle\n", label, ipc);
  const std::string unit = std::string(key).find("bounded_sequence") != std::string::npos
      ? "warp-sequences/nominal-cycle/SM" : "warp-PTX-ops/nominal-cycle/SM";
  g_results.log("instructions", key, ipc, unit);

  CUDA_CHECK(cudaFree(d_out));
  return ipc;
}

#define MEASURE_THP(name, init, label, key) \
  measure_inst_thp<NACC>(name(), init, label, key)

// -----------------------------------------------------------------------------
// FP32 arithmetic (use src==dst so the compiler cannot fold the loop and no
// extra operand-generation instructions are introduced).
struct InstFp32Fma {
  __device__ __forceinline__ void operator()(float &a, int iter) const {
    asm volatile("fma.rn.f32 %0,%0,%0,%0;" : "+f"(a));
  }
};
struct InstFp32Add {
  __device__ __forceinline__ void operator()(float &a, int iter) const {
    asm volatile("add.rn.f32 %0,%0,%0;" : "+f"(a));
  }
};
struct InstFp32Mul {
  __device__ __forceinline__ void operator()(float &a, int iter) const {
    asm volatile("mul.rn.f32 %0,%0,%0;" : "+f"(a));
  }
};

// INT32 arithmetic
struct InstInt32Add {
  __device__ __forceinline__ void operator()(int &a, int iter) const {
    asm volatile("add.s32 %0,%0,%1;" : "+r"(a) : "r"(threadIdx.x + 1));
  }
};
struct InstInt32Mul {
  __device__ __forceinline__ void operator()(int &a, int iter) const {
    asm volatile("mul.lo.s32 %0,%0,%1;" : "+r"(a) : "r"(threadIdx.x + 7));
  }
};

// FP64 arithmetic
struct InstFp64Add {
  __device__ __forceinline__ void operator()(double &a, int iter) const {
    asm volatile("add.rn.f64 %0,%0,%0;" : "+d"(a));
  }
};

// SFU/MUFU
struct InstRsqrt {
  __device__ __forceinline__ void operator()(float &a, int iter) const {
    asm volatile("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(a) : "f"(a));
  }
};
struct InstSin {
  __device__ __forceinline__ void operator()(float &a, int iter) const {
    asm volatile("sin.approx.ftz.f32 %0,%1;" : "=f"(a) : "f"(a));
  }
};
struct InstCos {
  __device__ __forceinline__ void operator()(float &a, int iter) const {
    asm volatile("cos.approx.ftz.f32 %0,%1;" : "=f"(a) : "f"(a));
  }
};
struct InstEx2 {
  __device__ __forceinline__ void operator()(float &a, int iter) const {
    a = __uint_as_float((__float_as_uint(a) & 0x007fffffU) | 0x3f000000U);
    asm volatile("ex2.approx.ftz.f32 %0,%1;" : "=f"(a) : "f"(a));
  }
};
struct InstLg2 {
  __device__ __forceinline__ void operator()(float &a, int iter) const {
    a = __uint_as_float((__float_as_uint(a) & 0x007fffffU) | 0x40000000U);
    asm volatile("lg2.approx.ftz.f32 %0,%1;" : "=f"(a) : "f"(a));
  }
};
struct InstRcp {
  __device__ __forceinline__ void operator()(float &a, int iter) const {
    asm volatile("rcp.approx.ftz.f32 %0,%1;" : "=f"(a) : "f"(a));
  }
};
struct InstTanh {
  __device__ __forceinline__ void operator()(float &a, int iter) const {
    asm volatile("tanh.approx.f32 %0,%1;" : "=f"(a) : "f"(a));
  }
};

// Extended integer
struct InstIadd3 {
  __device__ __forceinline__ void operator()(int &a, int iter) const {
    asm volatile("add.s32 %0,%0,%1;" : "+r"(a) : "r"(threadIdx.x + 1));
  }
};
struct InstImad {
  __device__ __forceinline__ void operator()(int &a, int iter) const {
    asm volatile("mad.lo.s32 %0,%0,%0,%0;" : "+r"(a));
  }
};
struct InstLea {
  __device__ __forceinline__ void operator()(int &a, int iter) const {
    asm volatile("mad.lo.s32 %0,%0,4,%1;" : "+r"(a) : "r"(threadIdx.x + 1));
  }
};
struct InstIdp4a {
  __device__ __forceinline__ void operator()(int &a, int iter) const {
    asm volatile("dp4a.s32.s32 %0,%0,%0,%1;" : "+r"(a) : "r"(0));
  }
};

// Logic / shift / permute
struct InstLop3 {
  __device__ __forceinline__ void operator()(unsigned int &a, int iter) const {
    // LUT 0x80 implements out = a when all three inputs are the same register.
    asm volatile("lop3.b32 %0,%0,%1,%2,0xca;" : "+r"(a) : "r"(threadIdx.x + 3U), "r"(threadIdx.x + 9U));
  }
};
struct InstShlConst {
  __device__ __forceinline__ void operator()(unsigned int &a, int iter) const {
    asm volatile("shl.b32 %0,%0,1;" : "+r"(a));
  }
};
struct InstShlVar {
  __device__ __forceinline__ void operator()(unsigned int &a, int iter) const {
    asm volatile("shl.b32 %0,%0,%1;" : "+r"(a) : "r"(threadIdx.x & 31));
  }
};
struct InstPrmt {
  __device__ __forceinline__ void operator()(unsigned int &a, int iter) const {
    asm volatile("prmt.b32 %0,%0,%0,0x1032;" : "+r"(a));
  }
};

// INT64 / DIV
struct InstDivU32 {
  __device__ __forceinline__ void operator()(unsigned int &a, int iter) const {
    unsigned int b = (unsigned int)(iter + 2);
    asm volatile("div.u32 %0,%1,%2;" : "=r"(a) : "r"(a), "r"(b));
  }
};
struct InstAddU64 {
  __device__ __forceinline__ void operator()(unsigned long long &a, int iter) const {
    asm volatile("add.u64 %0,%0,%0;" : "+l"(a));
  }
};
struct InstMulU64 {
  __device__ __forceinline__ void operator()(unsigned long long &a, int iter) const {
    asm volatile("mul.lo.u64 %0,%0,%0;" : "+l"(a));
  }
};

// Conversions: disabled in throughput mode because PTX register-type aliasing
// makes generic inline conversion chains fragile.  Latency numbers cover them.
struct InstCvtF32U32 {
  __device__ __forceinline__ void operator()(unsigned int &a, int iter) const {
    asm volatile(
        ".reg .f32 tmp;\n\t"
        "cvt.rn.f32.u32 tmp, %1;\n\t"
        "cvt.rn.u32.f32 %0, tmp;"
        : "+r"(a) : "r"(a));
  }
};
struct InstCvtF32S32 {
  __device__ __forceinline__ void operator()(int &a, int iter) const {
    asm volatile(
        ".reg .f32 tmp;\n\t"
        "cvt.rn.f32.s32 tmp, %1;\n\t"
        "cvt.rn.s32.f32 %0, tmp;"
        : "+r"(a) : "r"(a));
  }
};
struct InstCvtF32F64 {
  __device__ __forceinline__ void operator()(double &a, int iter) const {
    asm volatile(
        ".reg .f32 tmp;\n\t"
        "cvt.rn.f32.f64 tmp, %1;\n\t"
        "cvt.f64.f32 %0, tmp;"
        : "+d"(a) : "d"(a));
  }
};
struct InstCvtU64U32 {
  __device__ __forceinline__ void operator()(unsigned int &a, int iter) const {
    asm volatile(
        ".reg .u64 tmp;\n\t"
        "cvt.u64.u32 tmp, %1;\n\t"
        "cvt.u32.u64 %0, tmp;"
        : "+r"(a) : "r"(a));
  }
};

// SIMD packed
struct InstVadd2 {
  __device__ __forceinline__ void operator()(unsigned int &a, int iter) const {
    asm volatile("vadd2.u32.u32.u32 %0,%0,%0,%1;" : "+r"(a) : "r"(0));
  }
};
struct InstVadd4 {
  __device__ __forceinline__ void operator()(unsigned int &a, int iter) const {
    asm volatile("vadd4.u32.u32.u32 %0,%0,%0,%1;" : "+r"(a) : "r"(0));
  }
};
struct InstFmaF16x2 {
  __device__ __forceinline__ void operator()(unsigned int &c, int iter) const {
    asm volatile("fma.rn.f16x2 %0,%0,%0,%0;" : "+r"(c));
  }
};
struct InstFmaBf16x2 {
  __device__ __forceinline__ void operator()(unsigned int &c, int iter) const {
    asm volatile("fma.rn.bf16x2 %0,%0,%0,%0;" : "+r"(c));
  }
};
struct InstFmaF32x2 {
  __device__ __forceinline__ void operator()(unsigned long long &c, int iter) const {
    asm volatile("fma.rn.f32x2 %0,%0,%0,%0;" : "+l"(c));
  }
};

// Memory instruction throughput: LDG and STS counted as instructions/cycle.
// LDG uses inline-asm loads so the compiler cannot fold the loop, and four
// independent streams keep the load pipe busy.
__global__ void kernel_ldg_thp(const float *data, int partition, int iters,
                               float *out) {
  int tid = threadIdx.x;
  int mask = partition - 1;
  data += blockIdx.x * partition;
  int base = tid & mask;
  int idx0 = (base + 0) & mask;
  int idx1 = (base + 1) & mask;
  int idx2 = (base + 2) & mask;
  int idx3 = (base + 3) & mask;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  #pragma unroll 16
  for (int r = 0; r < iters; ++r) {
    idx0 = (idx0 + 4) & mask;
    idx1 = (idx1 + 4) & mask;
    idx2 = (idx2 + 4) & mask;
    idx3 = (idx3 + 4) & mask;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(acc0) : "l"(data + idx0));
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(acc1) : "l"(data + idx1));
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(acc2) : "l"(data + idx2));
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(acc3) : "l"(data + idx3));
  }
  out[blockIdx.x * blockDim.x + tid] = acc0 + acc1 + acc2 + acc3;
}

__global__ void kernel_sts_thp(int iters, float *out) {
  int tid = threadIdx.x;
  __shared__ float smem[1024];
  unsigned int addr = __cvta_generic_to_shared(smem + tid);
#pragma unroll 16
  for (int r = 0; r < iters; ++r) {
    asm volatile("st.volatile.shared.f32 [%0], %1;" :: "r"(addr), "f"(1.0f) : "memory");
    asm volatile("st.volatile.shared.f32 [%0], %1;" :: "r"(addr + 1024), "f"(2.0f) : "memory");
    asm volatile("st.volatile.shared.f32 [%0], %1;" :: "r"(addr + 2048), "f"(3.0f) : "memory");
    asm volatile("st.volatile.shared.f32 [%0], %1;" :: "r"(addr + 3072), "f"(4.0f) : "memory");
  }
  out[blockIdx.x * blockDim.x + tid] = smem[tid];
}

// -----------------------------------------------------------------------------
void benchmark_instruction_throughput() {
  printf("=== Instruction Throughput (inst/cycle per SM) ===\n");

  // FP32
  MEASURE_THP(InstFp32Fma, 1.0f, "FP32 FMA", "fp32_fma_throughput_ipc");
  MEASURE_THP(InstFp32Add, 1.0f, "FP32 ADD", "fp32_add_throughput_ipc");
  MEASURE_THP(InstFp32Mul, 1.0f, "FP32 MUL", "fp32_mul_throughput_ipc");
  // INT32
  MEASURE_THP(InstInt32Add, 1, "INT32 ADD", "int32_add_throughput_ipc");
  MEASURE_THP(InstInt32Mul, 1, "INT32 MUL", "int32_mul_throughput_ipc");
  // FP64
  MEASURE_THP(InstFp64Add, 1.0, "FP64 ADD", "fp64_add_throughput_ipc");
  // SFU
  MEASURE_THP(InstRsqrt, 1.0f, "RSQRT", "rsqrt_throughput_ipc");
  MEASURE_THP(InstSin, 1.0f, "SIN", "sin_throughput_ipc");
  MEASURE_THP(InstCos, 1.0f, "COS", "cos_throughput_ipc");
  MEASURE_THP(InstEx2, 1.0f, "EX2 + domain map", "ex2_bounded_sequence_throughput");
  MEASURE_THP(InstLg2, 1.0f, "LG2 + domain map", "lg2_bounded_sequence_throughput");
  MEASURE_THP(InstRcp, 1.0f, "RCP", "rcp_throughput_ipc");
  MEASURE_THP(InstTanh, 1.0f, "TANH", "tanh_throughput_ipc");
  // Extended integer
  MEASURE_THP(InstIadd3, 1, "IADD (add.s32)", "iadd_throughput_ipc");
  MEASURE_THP(InstImad, 1, "IMAD", "imad_throughput_ipc");
  MEASURE_THP(InstLea, 1, "address arithmetic", "address_arithmetic_throughput_ipc");
  MEASURE_THP(InstIdp4a, 0, "IDP4A", "idp4a_throughput_ipc");
  // Logic / shift / permute
  MEASURE_THP(InstLop3, 0xffffffffu, "LOP3", "lop3_throughput_ipc");
  MEASURE_THP(InstShlVar, 1u, "SHL var", "shl_var_throughput_ipc");
  MEASURE_THP(InstPrmt, 0u, "PRMT", "prmt_throughput_ipc");
  // INT64 / DIV
  MEASURE_THP(InstDivU32, 100u, "DIV u32", "div_u32_throughput_ipc");
  MEASURE_THP(InstAddU64, 1ULL, "ADD u64", "add_u64_throughput_ipc");
  MEASURE_THP(InstMulU64, 1ULL, "MUL u64", "mul_u64_throughput_ipc");
  // SIMD packed
  MEASURE_THP(InstVadd2, 0x00010001u, "VADD2", "vadd2_throughput_ipc");
  MEASURE_THP(InstVadd4, 0x01010101u, "VADD4", "vadd4_throughput_ipc");
  MEASURE_THP(InstFmaF16x2, 0x3c003c00u, "FMA f16x2", "fma_f16x2_throughput_ipc");
  MEASURE_THP(InstFmaBf16x2, 0x3f803f80u, "FMA bf16x2", "fma_bf16x2_throughput_ipc");
  MEASURE_THP(InstFmaF32x2, 0x3f8000003f800000ULL, "FMA f32x2", "fma_f32x2_throughput_ipc");

  // Memory throughput (LDG / STS)
  {
    DeviceInfo info = queryDevice(0);
    int threads = 256;
    int blocks = info.sm_count * 8;
    int iters = 1 << 14; // longer run for memory to get accurate timer
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));

    int partition = 512; // power of two, per-block L1 partition
    float *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, blocks * partition * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_buf, 0, blocks * partition * sizeof(float)));
    // LDG
    kernel_ldg_thp<<<blocks, threads>>>(d_buf, partition, iters, d_out);
    CUDA_CHECK(cudaGetLastError());
    GpuTimer ldg_timer;
    ldg_timer.start();
    kernel_ldg_thp<<<blocks, threads>>>(d_buf, partition, iters, d_out);
    ldg_timer.stop();
    CUDA_CHECK(cudaGetLastError());
    float ldg_ms = ldg_timer.elapsedMs();
    long long ldg_inst = (long long)blocks * (threads / 32) * 4 * iters;
    double ldg_ipc = (ldg_inst / (double)info.sm_count) /
                     ((ldg_ms * 1e-3) * ((double)info.clock_rate_khz * 1000.0));
    printf("  LDG throughput: %.2f inst/cycle\n", ldg_ipc);
    g_results.log("instructions", "ldg_throughput_ipc", ldg_ipc, "warp-PTX-ops/nominal-cycle/SM");

    // STS
    kernel_sts_thp<<<blocks, threads>>>(iters, d_out);
    CUDA_CHECK(cudaGetLastError());
    GpuTimer sts_timer;
    sts_timer.start();
    kernel_sts_thp<<<blocks, threads>>>(iters, d_out);
    sts_timer.stop();
    CUDA_CHECK(cudaGetLastError());
    float sts_ms = sts_timer.elapsedMs();
    long long sts_inst = (long long)blocks * (threads / 32) * 4 * iters;
    double sts_ipc = (sts_inst / (double)info.sm_count) /
                     ((sts_ms * 1e-3) * ((double)info.clock_rate_khz * 1000.0));
    printf("  STS throughput: %.2f inst/cycle\n", sts_ipc);
    g_results.log("instructions", "sts_throughput_ipc", sts_ipc, "warp-PTX-ops/nominal-cycle/SM");

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_out));
  }
}

} // namespace cmb
