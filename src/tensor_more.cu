#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

namespace cmb {

// TF32 MMA: m16n8k4, fragments A=2 f32, B=1 f32, C/D=4 f32 per thread.
__global__ void kernel_mma_m16n8k4_tf32_latency(unsigned int *cycles, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float a[2], b[1], c[4];
  #pragma unroll
  for (int i = 0; i < 2; ++i) a[i] = 1.0f;
  b[0] = 1.0f;
  for (int i = 0; i < 4; ++i) c[i] = 0.0f;

  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "f"(a[0]), "f"(a[1]),
          "f"(b[0]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
  CMB_CLOCK_STOP();

  CMB_DEPENDENCY_GUARD(c[0]);
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
#endif
}

__global__ void kernel_mma_m16n8k4_tf32_throughput(float *out, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float a[2], b[1], c[4];
  #pragma unroll
  for (int i = 0; i < 2; ++i) a[i] = 1.0f;
  b[0] = 1.0f;
  for (int i = 0; i < 4; ++i) c[i] = 0.0f;

  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "f"(a[0]), "f"(a[1]),
          "f"(b[0]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
  float acc = c[0] + c[1] + c[2] + c[3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#endif
}

// FP64 DMMA: m8n8k4, one double per A/B, two doubles per C/D thread.
__global__ void kernel_dmma_f64_latency(unsigned int *cycles, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  double a = 1.0;
  double b = 1.0;
  double c[2] = {0.0, 0.0};

  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0,%1}, {%2}, {%3}, {%4,%5};"
        : "=d"(c[0]), "=d"(c[1])
        : "d"(a), "d"(b), "d"(c[0]), "d"(c[1]));
  }
  CMB_CLOCK_STOP();

  CMB_DEPENDENCY_GUARD(c[0]);
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
#endif
}

__global__ void kernel_dmma_f64_throughput(double *out, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  double a = 1.0;
  double b = 1.0;
  double c[2] = {0.0, 0.0};

  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0,%1}, {%2}, {%3}, {%4,%5};"
        : "=d"(c[0]), "=d"(c[1])
        : "d"(a), "d"(b), "d"(c[0]), "d"(c[1]));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = c[0] + c[1];
#endif
}

// Pipe overlap: FFMA-only chain (per-thread), measured in same config as MMA.
__global__ void kernel_fma_only_for_overlap(unsigned int *cycles, int iters) {
  float fa = 1.0f + threadIdx.x * 0.001f, fb = 1.0001f;
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (threadIdx.x == 0)
    s_t0 = clockVal();
  __syncthreads();
  if (threadIdx.x < 32) {
    for (int i = 0; i < iters; ++i) {
      float fr;
      asm volatile("fma.rn.f32 %0,%1,%2,%3;"
                   : "=f"(fr) : "f"(fa), "f"(fb), "f"(fa));
      fa = fr;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    s_t1 = clockVal();
  __syncthreads();
  if (threadIdx.x == 0)
    cycles[0] = s_t1 - s_t0;
  if (fa == 0xdeadbeef)
    cycles[0] = 0;
}

// MMA-only chain (warp-collective), m16n8k8 TF32.
__global__ void kernel_mma_only_for_overlap(unsigned int *cycles, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float a[4], b[2], c[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 1.0f;
  for (int i = 0; i < 2; ++i) b[i] = 1.0f;
  for (int i = 0; i < 4; ++i) c[i] = 0.0f;

  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (threadIdx.x == 0)
    s_t0 = clockVal();
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
  __syncthreads();
  if (threadIdx.x == 0)
    s_t1 = clockVal();
  __syncthreads();
  if (threadIdx.x == 0)
    cycles[0] = s_t1 - s_t0;
  if ((int)c[0] == 0xdeadbeef)
    cycles[0] = 0;
#endif
}

// Interleaved FFMA + MMA dependent chains.
__global__ void kernel_fma_mma_mixed(unsigned int *cycles, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float fa = 1.0f + threadIdx.x * 0.001f, fb = 1.0001f;
  float a[4], b[2], c[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 1.0f;
  for (int i = 0; i < 2; ++i) b[i] = 1.0f;
  for (int i = 0; i < 4; ++i) c[i] = 0.0f;

  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (threadIdx.x == 0)
    s_t0 = clockVal();
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    float fr;
    asm volatile("fma.rn.f32 %0,%1,%2,%3;"
                 : "=f"(fr) : "f"(fa), "f"(fb), "f"(fa));
    fa = fr;
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
  __syncthreads();
  if (threadIdx.x == 0)
    s_t1 = clockVal();
  __syncthreads();
  if (threadIdx.x == 0)
    cycles[0] = s_t1 - s_t0;
  if (fa == 0xdeadbeef || (int)c[0] == 0xdeadbeef)
    cycles[0] = 0;
#endif
}

void benchmark_tensor_more() {
  DeviceInfo info = queryDevice(0);
  printf("=== Tensor Core (More Shapes) ===\n");

  int cc = info.cc_major * 10 + info.cc_minor;
  if (cc < 80) {
    printf("  Tensor cores require compute capability >= 8.0. Skipping.\n");
    return;
  }

  // FP64 DMMA latency
  {
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    int iters = 1 << 12;
    kernel_dmma_f64_latency<<<1, 32>>>(d_cycles, iters);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h_cycles;
    CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double cyc = (double)h_cycles / iters;
    double ns = cyclesToNominalNs(cyc, info.clock_rate_khz);
    printf("  DMMA m8n8k4 (FP64) latency: %.2f cycles (%.2f nominal ns)\n", cyc, ns);
    g_results.log("tensor_core_more", "dmma_m8n8k4_latency_cyc", cyc, "cycles");
    g_results.log("tensor_core_more", "dmma_m8n8k4_latency_ns", ns, "nominal ns");
    CUDA_CHECK(cudaFree(d_cycles));
  }

  // FP64 DMMA throughput
  {
    int iters = 1 << 12;
    int threads = 32;
    int blocks = info.sm_count * info.max_warps_per_sm;
    double *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(double)));

    GpuTimer timer;
    timer.start();
    kernel_dmma_f64_throughput<<<blocks, threads>>>(d_out, iters);
    timer.stop();
    float ms = timer.elapsedMs();
    CUDA_CHECK(cudaGetLastError());

    // 8*8*4 = 256 DFMAs per mma, each = 2 flops
    long long ops = (long long)blocks * threads / 32 * iters * 8 * 8 * 4 * 2;
    double tflops = ops / (ms * 1e-3) / 1e12;
    printf("  DMMA m8n8k4 (FP64) throughput: %.3f TFLOP/s\n", tflops);
    g_results.log("tensor_core_more", "dmma_m8n8k4_throughput_tflops", tflops,
                  "TFLOP/s");
    CUDA_CHECK(cudaFree(d_out));
  }

  // TF32 m16n8k4 latency
  {
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    int iters = 1 << 12;
    kernel_mma_m16n8k4_tf32_latency<<<1, 32>>>(d_cycles, iters);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h_cycles;
    CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double cyc = (double)h_cycles / iters;
    double ns = cyclesToNominalNs(cyc, info.clock_rate_khz);
    printf("  MMA m16n8k4 (TF32) latency: %.2f cycles (%.2f nominal ns)\n", cyc, ns);
    g_results.log("tensor_core_more", "mma_m16n8k4_tf32_latency_cyc", cyc, "cycles");
    g_results.log("tensor_core_more", "mma_m16n8k4_tf32_latency_ns", ns, "nominal ns");
    CUDA_CHECK(cudaFree(d_cycles));
  }

  // TF32 m16n8k4 throughput
  {
    int iters = 1 << 12;
    int threads = 32;
    int blocks = info.sm_count * info.max_warps_per_sm;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));

    GpuTimer timer;
    timer.start();
    kernel_mma_m16n8k4_tf32_throughput<<<blocks, threads>>>(d_out, iters);
    timer.stop();
    float ms = timer.elapsedMs();
    CUDA_CHECK(cudaGetLastError());

    // 16*8*4 = 512 FMAs per mma, each = 2 flops
    long long ops = (long long)blocks * threads / 32 * iters * 16 * 8 * 4 * 2;
    double tflops = ops / (ms * 1e-3) / 1e12;
    printf("  MMA m16n8k4 (TF32) throughput: %.2f TFLOP/s\n", tflops);
    g_results.log("tensor_core_more", "mma_m16n8k4_tf32_throughput_tflops", tflops,
                  "TFLOP/s");
    CUDA_CHECK(cudaFree(d_out));
  }

  // MMA + FFMA overlap ratio
  {
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    int iters = 1 << 12;

    kernel_fma_only_for_overlap<<<1, 32>>>(d_cycles, iters);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h;
    CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double fma_cyc = (double)h / iters;

    kernel_mma_only_for_overlap<<<1, 32>>>(d_cycles, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double mma_cyc = (double)h / iters;

    kernel_fma_mma_mixed<<<1, 32>>>(d_cycles, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double mixed_cyc = (double)h / iters;

    double overlap = (fma_cyc + mma_cyc - mixed_cyc) /
                     std::min(fma_cyc, mma_cyc);
    if (overlap < 0.0)
      overlap = 0.0;
    if (overlap > 1.0)
      overlap = 1.0;
    printf("  FMA+TFMA overlap: FMA=%.2f cyc, MMA=%.2f cyc, mixed=%.2f cyc, "
           "overlap=%.2f%%\n",
           fma_cyc, mma_cyc, mixed_cyc, overlap * 100.0);
    g_results.log("tensor_core_more", "mma_ffma_overlap_ratio", overlap, "");
    CUDA_CHECK(cudaFree(d_cycles));
  }

  // ISA availability is distinct from whether this suite has a benchmark.
  // PTX documents WGMMA only for sm_90a; sm_120a/f supports FP6/FP4 MMA.
  if (cc == 120) {
    g_results.log("tensor_core_more", "wgmma_supported", 0.0, "bool (PTX target)");
    g_results.log("tensor_core_more", "fp6_fp4_mma_supported", 1.0, "bool (requires sm_120a/f)");
  }
  g_results.log("tensor_core_more", "fp6_fp4_benchmark_implemented", 0.0, "bool");
}

} // namespace cmb
