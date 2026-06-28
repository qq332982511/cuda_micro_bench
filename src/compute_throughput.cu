#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

// Throughput measured with independent PTX instruction chains (ILP=4).
// Operands depend on threadIdx to prevent ptxas from strength-reducing loops.

__global__ void kernel_fp32_add(float *out, int iters) {
  float a0 = 1.0f, a1 = 2.0f, a2 = 3.0f, a3 = 4.0f;
  float b0 = 1.0001f + threadIdx.x;
  float b1 = 1.0002f + threadIdx.x;
  float b2 = 1.0003f + threadIdx.x;
  float b3 = 1.0004f + threadIdx.x;
  for (int i = 0; i < iters; ++i) {
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a0) : "f"(b0));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a1) : "f"(b1));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a2) : "f"(b2));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a3) : "f"(b3));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_fp32_mul(float *out, int iters) {
  float a0 = 1.0f, a1 = 2.0f, a2 = 3.0f, a3 = 4.0f;
  float b0 = 1.0001f + threadIdx.x;
  float b1 = 1.0002f + threadIdx.x;
  float b2 = 1.0003f + threadIdx.x;
  float b3 = 1.0004f + threadIdx.x;
  for (int i = 0; i < iters; ++i) {
    asm volatile("mul.f32 %0, %0, %1;" : "+f"(a0) : "f"(b0));
    asm volatile("mul.f32 %0, %0, %1;" : "+f"(a1) : "f"(b1));
    asm volatile("mul.f32 %0, %0, %1;" : "+f"(a2) : "f"(b2));
    asm volatile("mul.f32 %0, %0, %1;" : "+f"(a3) : "f"(b3));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_fp32_fma(float *out, int iters) {
  float a0 = 1.0f, a1 = 2.0f, a2 = 3.0f, a3 = 4.0f;
  float b = 1.0001f + threadIdx.x;
  float c = 0.5f;
  for (int i = 0; i < iters; ++i) {
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a0) : "f"(b), "f"(c));
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a1) : "f"(b), "f"(c));
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a2) : "f"(b), "f"(c));
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a3) : "f"(b), "f"(c));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_int32_add(int *out, int iters) {
  int a0 = 1, a1 = 2, a2 = 3, a3 = 4;
  int b0 = 1 + threadIdx.x;
  int b1 = 2 + threadIdx.x;
  int b2 = 3 + threadIdx.x;
  int b3 = 4 + threadIdx.x;
  for (int i = 0; i < iters; ++i) {
    asm volatile("add.s32 %0, %0, %1;" : "+r"(a0) : "r"(b0));
    asm volatile("add.s32 %0, %0, %1;" : "+r"(a1) : "r"(b1));
    asm volatile("add.s32 %0, %0, %1;" : "+r"(a2) : "r"(b2));
    asm volatile("add.s32 %0, %0, %1;" : "+r"(a3) : "r"(b3));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_int32_mul(int *out, int iters) {
  int a0 = 2, a1 = 3, a2 = 4, a3 = 5;
  int b0 = 7 + threadIdx.x;
  int b1 = 8 + threadIdx.x;
  int b2 = 9 + threadIdx.x;
  int b3 = 10 + threadIdx.x;
  for (int i = 0; i < iters; ++i) {
    asm volatile("mul.lo.s32 %0, %0, %1;" : "+r"(a0) : "r"(b0));
    asm volatile("mul.lo.s32 %0, %0, %1;" : "+r"(a1) : "r"(b1));
    asm volatile("mul.lo.s32 %0, %0, %1;" : "+r"(a2) : "r"(b2));
    asm volatile("mul.lo.s32 %0, %0, %1;" : "+r"(a3) : "r"(b3));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_fp64_add(double *out, int iters) {
  double a0 = 1.0, a1 = 1.0, a2 = 1.0, a3 = 1.0;
  double b0 = 1.0001 + threadIdx.x;
  double b1 = 1.0002 + threadIdx.x;
  double b2 = 1.0003 + threadIdx.x;
  double b3 = 1.0004 + threadIdx.x;
  for (int i = 0; i < iters; ++i) {
    asm volatile("add.f64 %0, %0, %1;" : "+d"(a0) : "d"(b0));
    asm volatile("add.f64 %0, %0, %1;" : "+d"(a1) : "d"(b1));
    asm volatile("add.f64 %0, %0, %1;" : "+d"(a2) : "d"(b2));
    asm volatile("add.f64 %0, %0, %1;" : "+d"(a3) : "d"(b3));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_sfu_rsqrt(float *out, int iters) {
  float a0 = 1.0f, a1 = 2.0f, a2 = 3.0f, a3 = 4.0f;
  for (int i = 0; i < iters; ++i) {
    a0 = rsqrt(a0);
    a1 = rsqrt(a1);
    a2 = rsqrt(a2);
    a3 = rsqrt(a3);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_sfu_sin(float *out, int iters) {
  float a0 = 0.1f, a1 = 0.2f, a2 = 0.3f, a3 = 0.4f;
  for (int i = 0; i < iters; ++i) {
    a0 = sin(a0);
    a1 = sin(a1);
    a2 = sin(a2);
    a3 = sin(a3);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

static double measure_fp32_throughput(void (*kernel)(float *, int),
                                      const char *label, int iters) {
  DeviceInfo info = queryDevice(0);
  int threads = 256;
  int blocks = info.sm_count * 8;
  float *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));

  kernel<<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK(cudaGetLastError());

  GpuTimer timer;
  timer.start();
  kernel<<<blocks, threads>>>(d_out, iters);
  timer.stop();
  float ms = timer.elapsedMs();
  CUDA_CHECK(cudaGetLastError());

  long long ops = (long long)blocks * threads * iters * 4;
  double gflops = ops / (ms * 1e-3) / 1e9;
  printf("  %-15s: %.2f GFLOP/s\n", label, gflops);

  CUDA_CHECK(cudaFree(d_out));
  return gflops;
}

template <typename T>
static double measure_int_throughput(void (*kernel)(T *, int),
                                     const char *label, int iters) {
  DeviceInfo info = queryDevice(0);
  int threads = 256;
  int blocks = info.sm_count * 8;
  T *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(T)));

  kernel<<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK(cudaGetLastError());

  GpuTimer timer;
  timer.start();
  kernel<<<blocks, threads>>>(d_out, iters);
  timer.stop();
  float ms = timer.elapsedMs();
  CUDA_CHECK(cudaGetLastError());

  long long ops = (long long)blocks * threads * iters * 4;
  double gops = ops / (ms * 1e-3) / 1e9;
  printf("  %-15s: %.2f GOPS/s\n", label, gops);

  CUDA_CHECK(cudaFree(d_out));
  return gops;
}

void benchmark_compute_throughput() {
  printf("=== Compute Throughput ===\n");
  int iters = 1 << 12;
  double fp32_add = measure_fp32_throughput(kernel_fp32_add, "FP32 ADD", iters);
  double fp32_mul = measure_fp32_throughput(kernel_fp32_mul, "FP32 MUL", iters);
  double fp32_fma = measure_fp32_throughput(kernel_fp32_fma, "FP32 FMA", iters);
  double int32_add = measure_int_throughput(kernel_int32_add, "INT32 ADD", iters);
  double int32_mul = measure_int_throughput(kernel_int32_mul, "INT32 MUL", iters);
  double fp64_add = measure_int_throughput(kernel_fp64_add, "FP64 ADD", iters);
  g_results.log("instructions", "fp32_add_gflops", fp32_add, "GFLOP/s");
  g_results.log("instructions", "fp32_mul_gflops", fp32_mul, "GFLOP/s");
  g_results.log("instructions", "fp32_fma_gflops", fp32_fma, "GFLOP/s");
  g_results.log("instructions", "int32_add_gops", int32_add, "GOPS/s");
  g_results.log("instructions", "int32_mul_gops", int32_mul, "GOPS/s");
  g_results.log("instructions", "fp64_add_gflops", fp64_add, "GFLOP/s");
  // Alias used by the HTML report/reference comparison (FMA = 2 FLOPs)
  g_results.log("peak", "fp32_cuda_cores_peak_tflops", fp32_fma * 2.0 / 1000.0,
                "TFLOP/s");

  {
    DeviceInfo info = queryDevice(0);
    int threads = 256;
    int blocks = info.sm_count * 4;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
    kernel_sfu_rsqrt<<<blocks, threads>>>(d_out, iters);
    GpuTimer timer;
    timer.start();
    kernel_sfu_rsqrt<<<blocks, threads>>>(d_out, iters);
    timer.stop();
    float ms = timer.elapsedMs();
    long long ops = (long long)blocks * threads * iters * 4;
    double rsqrt_gops = ops / (ms * 1e-3) / 1e9;
    printf("  %-15s: %.2f GOP/s\n", "RSQRT", rsqrt_gops);
    g_results.log("instructions", "rsqrt_gops", rsqrt_gops, "GOPS/s");
    CUDA_CHECK(cudaFree(d_out));
  }
  {
    DeviceInfo info = queryDevice(0);
    int threads = 256;
    int blocks = info.sm_count * 4;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
    kernel_sfu_sin<<<blocks, threads>>>(d_out, iters);
    GpuTimer timer;
    timer.start();
    kernel_sfu_sin<<<blocks, threads>>>(d_out, iters);
    timer.stop();
    float ms = timer.elapsedMs();
    long long ops = (long long)blocks * threads * iters * 4;
    double sin_gops = ops / (ms * 1e-3) / 1e9;
    printf("  %-15s: %.2f GOP/s\n", "SIN", sin_gops);
    g_results.log("instructions", "sin_gops", sin_gops, "GOPS/s");
    CUDA_CHECK(cudaFree(d_out));
  }
}

} // namespace cmb
