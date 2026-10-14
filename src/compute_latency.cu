#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

__global__ void kernel_latency_fp32_add(unsigned int *cycles, int iters) {
  float a = 1.0f + threadIdx.x * 0.001f;
  float b = 1.0001f;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    float r;
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_fp32_mul(unsigned int *cycles, int iters) {
  float a = 1.001f + threadIdx.x * 0.001f;
  float b = 1.0001f;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    float r;
    asm volatile("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_fp32_fma(unsigned int *cycles, int iters) {
  float a = 1.0f + threadIdx.x * 0.001f;
  float b = 1.0001f;
  float c = 0.5f;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    float r;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                 : "=f"(r)
                 : "f"(a), "f"(b), "f"(c));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

// Alternating two-register chain keeps intermediate results live and prevents
// repeated ADDs from merging into IADD3 or a scaled loop increment.
__global__ void kernel_latency_int32_add(unsigned int *cycles, int iters) {
  int a = 1 + threadIdx.x, b = 2 + threadIdx.x;
  CMB_CLOCK_START();
#pragma unroll 16
  for (int i = 0; i < iters / 2; ++i) {
    asm volatile("add.s32 %0, %0, %1;\n\tadd.s32 %1, %0, %1;"
                 : "+r"(a), "+r"(b));
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a ^ b);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_int32_mul(unsigned int *cycles, int iters) {
  int a = 2 + threadIdx.x;
  int b = 7 + threadIdx.x;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    int r;
    asm volatile("mul.lo.s32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_fp64_add(unsigned int *cycles, int iters) {
  double a = 1.0 + threadIdx.x * 0.001;
  double b = 1.0001;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    double r;
    asm volatile("add.f64 %0, %1, %2;" : "=d"(r) : "d"(a), "d"(b));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_fp64_fma(unsigned int *cycles, int iters) {
  double a = 1.0 + threadIdx.x * 0.001;
  double b = 1.0001;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    double r;
    asm volatile("fma.rn.f64 %0, %1, %2, %3;" : "=d"(r) : "d"(a), "d"(b), "d"(0.5));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_rsqrt(unsigned int *cycles, int iters) {
  float a = 1.0f + threadIdx.x * 0.001f;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    float r;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_sin(unsigned int *cycles, int iters) {
  float a = 0.5f + threadIdx.x * 0.001f;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    float r;
    asm volatile("sin.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_cos(unsigned int *cycles, int iters) {
  float a = 0.5f + threadIdx.x * 0.001f;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    float r;
    asm volatile("cos.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_ex2(unsigned int *cycles, int iters) {
  float a = 1.0f + threadIdx.x * 0.001f;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    a = __uint_as_float((__float_as_uint(a) & 0x007fffffU) | 0x3f000000U);
    float r;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_lg2(unsigned int *cycles, int iters) {
  float a = 2.0f + threadIdx.x * 0.001f;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    a = __uint_as_float((__float_as_uint(a) & 0x007fffffU) | 0x40000000U);
    float r;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_rcp(unsigned int *cycles, int iters) {
  float a = 1.0001f + threadIdx.x * 0.001f;
  CMB_CLOCK_START();
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    float r;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

__global__ void kernel_latency_tanh(unsigned int *cycles, int iters) {
  float a = 0.5f + threadIdx.x * 0.001f;
  CMB_CLOCK_START();
  #pragma unroll 32
  for (int i = 0; i < iters; ++i) {
    float r;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    a = r;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  cycles[0] = t1 - t0;
}

static double measure_latency(void (*kernel)(unsigned int *, int),
                              const char *label, int iters) {
  DeviceInfo info = queryDevice(0);
  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));

  kernel<<<1, 1>>>(d_cycles, iters);
  CUDA_CHECK(cudaGetLastError());

  unsigned int h_cycles;
  CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  double cyc = (double)h_cycles / iters;
  double ns = cyclesToNominalNs(cyc, info.clock_rate_khz);
  printf("  %-15s: %.2f cycles (%.2f nominal ns)\n", label, cyc, ns);

  CUDA_CHECK(cudaFree(d_cycles));
  return cyc;
}

void benchmark_compute_latency() {
  printf("=== Compute Latency ===\n");
  int iters = 1 << 14;
  double fp32_add = measure_latency(kernel_latency_fp32_add, "FP32 ADD", iters);
  double fp32_mul = measure_latency(kernel_latency_fp32_mul, "FP32 MUL", iters);
  double fp32_fma = measure_latency(kernel_latency_fp32_fma, "FP32 FMA", iters);
  double int32_add = measure_latency(kernel_latency_int32_add, "INT32 ADD", iters);
  double int32_mul = measure_latency(kernel_latency_int32_mul, "INT32 MUL", iters);
  double fp64_add = measure_latency(kernel_latency_fp64_add, "FP64 ADD", iters);
  double fp64_fma = measure_latency(kernel_latency_fp64_fma, "FP64 FMA", iters);
  double rsqrt = measure_latency(kernel_latency_rsqrt, "RSQRT", iters);
  double sin_lat = measure_latency(kernel_latency_sin, "SIN", iters);
  double cos_lat = measure_latency(kernel_latency_cos, "COS", iters);
  double ex2_lat = measure_latency(kernel_latency_ex2, "EX2 + domain map", iters);
  double lg2_lat = measure_latency(kernel_latency_lg2, "LG2 + domain map", iters);
  double rcp_lat = measure_latency(kernel_latency_rcp, "RCP", iters);
  double tanh_lat = measure_latency(kernel_latency_tanh, "TANH", iters);
  g_results.log("instructions", "fp32_add_latency_cyc", fp32_add, "cycles");
  g_results.log("instructions", "fp32_mul_latency_cyc", fp32_mul, "cycles");
  g_results.log("instructions", "fp32_fma_latency_cyc", fp32_fma, "cycles");
  g_results.log("instructions", "int32_add_latency_cyc", int32_add, "cycles");
  g_results.log("instructions", "int32_mul_latency_cyc", int32_mul, "cycles");
  g_results.log("instructions", "fp64_add_latency_cyc", fp64_add, "cycles");
  g_results.log("instructions", "rsqrt_latency_cyc", rsqrt, "cycles");
  g_results.log("instructions", "sin_latency_cyc", sin_lat, "cycles");
  g_results.log("instructions", "cos_latency_cyc", cos_lat, "cycles");
  g_results.log("instructions", "ex2_bounded_chain_cyc", ex2_lat, "cycles");
  g_results.log("instructions", "lg2_bounded_chain_cyc", lg2_lat, "cycles");
  g_results.log("instructions", "rcp_latency_cyc", rcp_lat, "cycles");
  g_results.log("instructions", "tanh_latency_cyc", tanh_lat, "cycles");
  // Alias keys used by the HTML report/reference comparison
  g_results.log("instructions", "fadd_latency_cyc", fp32_add, "cycles");
  g_results.log("instructions", "ffma_latency_cyc", fp32_fma, "cycles");
  g_results.log("instructions", "dfma_latency_cyc", fp64_fma, "cycles");
  g_results.log("instructions", "mufu_rsq_latency_cyc", rsqrt, "cycles");
}

} // namespace cmb
