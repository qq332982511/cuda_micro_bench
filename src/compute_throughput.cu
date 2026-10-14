#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <type_traits>

namespace cmb {

// Throughput measured with independent PTX instruction chains (ILP=4).
// Operands depend on threadIdx to prevent ptxas from strength-reducing loops.

__global__ void kernel_fp32_add(float *out, int iters) {
  float a0 = 1.0f + threadIdx.x * 0.001f, a1 = 2.0f, a2 = 3.0f, a3 = 4.0f;
  float b0 = 1.0001f + threadIdx.x;
  float b1 = 1.0002f + threadIdx.x;
  float b2 = 1.0003f + threadIdx.x;
  float b3 = 1.0004f + threadIdx.x;
  #pragma unroll 16
  for (int i = 0; i < iters; ++i) {
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a0) : "f"(b0));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a1) : "f"(b1));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a2) : "f"(b2));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(a3) : "f"(b3));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_fp32_mul(float *out, int iters) {
  float a0 = 1.0f + threadIdx.x * 0.001f, a1 = 2.0f, a2 = 3.0f, a3 = 4.0f;
  float b0 = 1.0001f + threadIdx.x;
  float b1 = 1.0002f + threadIdx.x;
  float b2 = 1.0003f + threadIdx.x;
  float b3 = 1.0004f + threadIdx.x;
  #pragma unroll 16
  for (int i = 0; i < iters; ++i) {
    asm volatile("mul.f32 %0, %0, %1;" : "+f"(a0) : "f"(b0));
    asm volatile("mul.f32 %0, %0, %1;" : "+f"(a1) : "f"(b1));
    asm volatile("mul.f32 %0, %0, %1;" : "+f"(a2) : "f"(b2));
    asm volatile("mul.f32 %0, %0, %1;" : "+f"(a3) : "f"(b3));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_fp32_fma(float *out, int iters) {
  float a0 = 1.0f + threadIdx.x * 0.001f, a1 = 2.0f, a2 = 3.0f, a3 = 4.0f;
  float b = 1.0001f + threadIdx.x;
  float c = 0.5f;
  #pragma unroll 16
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
  #pragma unroll 1
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
  #pragma unroll 16
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
  #pragma unroll 16
  for (int i = 0; i < iters; ++i) {
    asm volatile("add.f64 %0, %0, %1;" : "+d"(a0) : "d"(b0));
    asm volatile("add.f64 %0, %0, %1;" : "+d"(a1) : "d"(b1));
    asm volatile("add.f64 %0, %0, %1;" : "+d"(a2) : "d"(b2));
    asm volatile("add.f64 %0, %0, %1;" : "+d"(a3) : "d"(b3));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_sfu_rsqrt(float *out, int iters) {
  float a0 = 1.0f + threadIdx.x * 0.001f, a1 = 2.0f, a2 = 3.0f, a3 = 4.0f;
  #pragma unroll 16
  for (int i = 0; i < iters; ++i) {
    asm volatile("rsqrt.approx.ftz.f32 %0,%0;" : "+f"(a0));
    asm volatile("rsqrt.approx.ftz.f32 %0,%0;" : "+f"(a1));
    asm volatile("rsqrt.approx.ftz.f32 %0,%0;" : "+f"(a2));
    asm volatile("rsqrt.approx.ftz.f32 %0,%0;" : "+f"(a3));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void kernel_sfu_sin(float *out, int iters) {
  float a0 = 0.1f, a1 = 0.2f, a2 = 0.3f, a3 = 0.4f;
  #pragma unroll 16
  for (int i = 0; i < iters; ++i) {
    asm volatile("sin.approx.ftz.f32 %0,%0;" : "+f"(a0));
    asm volatile("sin.approx.ftz.f32 %0,%0;" : "+f"(a1));
    asm volatile("sin.approx.ftz.f32 %0,%0;" : "+f"(a2));
    asm volatile("sin.approx.ftz.f32 %0,%0;" : "+f"(a3));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

// Warm with sustained work after serial-memory sections, without changing
// system-wide clocks. Sample the same kernel repeatedly to reduce WDDM noise.
template <typename Kernel, typename T>
static double warm_and_measure(Kernel kernel, T *out, int blocks, int threads, int iters) {
  double warm_ms = 0.0;
  do {
    GpuTimer timer;
    timer.start();
    kernel<<<blocks, threads>>>(out, iters * 16);
    CUDA_CHECK(cudaGetLastError());
    timer.stop();
    warm_ms += timer.elapsedMs();
  } while (warm_ms < 200.0);
  std::vector<double> samples;
  for (int sample = 0; sample < 5; ++sample) {
    GpuTimer timer;
    timer.start();
    kernel<<<blocks, threads>>>(out, iters);
    CUDA_CHECK(cudaGetLastError());
    timer.stop();
    samples.push_back(timer.elapsedMs());
  }
  std::sort(samples.begin(), samples.end());
  return samples[2];
}

static double measure_fp32_throughput(void (*kernel)(float *, int),
                                      const char *label, int iters, int flops_per_inst = 1,
                                      const char *unit = "GFLOP/s", int blocks_per_sm = 8) {
  DeviceInfo info = queryDevice(0);
  int threads = 256;
  int blocks = info.sm_count * blocks_per_sm;
  float *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));

  double ms = warm_and_measure(kernel, d_out, blocks, threads, iters);

  long long ops = (long long)blocks * threads * iters * 4;
  double gflops = ops * flops_per_inst / (ms * 1e-3) / 1e9;
  printf("  %-15s: %.2f %s\n", label, gflops, unit);

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

  double ms = warm_and_measure(kernel, d_out, blocks, threads, iters);

  long long ops = (long long)blocks * threads * iters * 4;
  double gops = ops / (ms * 1e-3) / 1e9;
  printf("  %-15s: %.2f %s\n", label, gops,
         std::is_floating_point<T>::value ? "GFLOP/s" : "GOP/s");

  CUDA_CHECK(cudaFree(d_out));
  return gops;
}

void benchmark_compute_throughput() {
  printf("=== Compute Throughput ===\n");
  int iters = 1 << 12;
  double fp32_add = measure_fp32_throughput(kernel_fp32_add, "FP32 ADD", iters);
  double fp32_mul = measure_fp32_throughput(kernel_fp32_mul, "FP32 MUL", iters);
  double fp32_fma = measure_fp32_throughput(kernel_fp32_fma, "FP32 FMA", iters, 2);
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
  g_results.log("peak", "fp32_cuda_cores_peak_tflops", fp32_fma / 1000.0,
                "TFLOP/s");

  double rsqrt_gops = measure_fp32_throughput(kernel_sfu_rsqrt, "RSQRT", iters, 1, "GOP/s", 4);
  double sin_gops = measure_fp32_throughput(kernel_sfu_sin, "SIN", iters, 1, "GOP/s", 4);
  g_results.log("instructions", "rsqrt_gops", rsqrt_gops, "GOP/s");
  g_results.log("instructions", "sin_gops", sin_gops, "GOP/s");
}

} // namespace cmb
