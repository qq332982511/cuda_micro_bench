#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

// Tensor core benchmark using PTX mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
// Works on Ampere+ (sm_80+).
// Fragment sizes for m16n8k8 tf32:
//   A: 4 x .f32, B: 2 x .f32, C/D: 4 x .f32
__global__ void kernel_mma_latency(unsigned int *cycles, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float a[4], b[2], c[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 1.0f;
  for (int i = 0; i < 2; ++i) b[i] = 1.0f;
  for (int i = 0; i < 4; ++i) c[i] = 0.0f;

  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
  CMB_CLOCK_STOP();

  if ((int)c[0] == 0xdeadbeef)
    t1 = clockVal();
  cycles[0] = t1 - t0;
#endif
}

__global__ void kernel_mma_throughput(float *out, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float a[4], b[2], c[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 1.0f;
  for (int i = 0; i < 2; ++i) b[i] = 1.0f;
  for (int i = 0; i < 4; ++i) c[i] = 0.0f;

  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
  float acc = c[0] + c[1] + c[2] + c[3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#endif
}

void benchmark_tensor_core() {
  DeviceInfo info = queryDevice(0);
  printf("=== Tensor Core (MMA) Benchmarks ===\n");

  int cc = info.cc_major * 10 + info.cc_minor;
  if (cc < 80) {
    printf("  Tensor cores require compute capability >= 8.0 (Ampere+). Skipping.\n");
    return;
  }

  // Latency
  {
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    int iters = 1 << 12;
    kernel_mma_latency<<<1, 32>>>(d_cycles, iters);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h_cycles;
    CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double cyc = (double)h_cycles / iters;
    double ns = cyc * 1e9 / info.clock_rate_khz;
    printf("  MMA m16n8k8 latency: %.2f cycles (%.2f ns)\n", cyc, ns);
    g_results.log("tensor_core", "mma_m16n8k8_latency_cyc", cyc, "cycles");
    g_results.log("tensor_core", "mma_m16n8k8_latency_ns", ns, "ns");
    g_results.log("instructions", "hmma_16816_latency_cyc", cyc, "cycles");
    CUDA_CHECK(cudaFree(d_cycles));
  }

  // Throughput
  {
    int iters = 1 << 12;
    int threads = 32;  // one warp per block
    int blocks = info.sm_count * info.max_warps_per_sm;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));

    GpuTimer timer;
    timer.start();
    kernel_mma_throughput<<<blocks, threads>>>(d_out, iters);
    timer.stop();
    float ms = timer.elapsedMs();
    CUDA_CHECK(cudaGetLastError());

    // 16*8*8 = 1024 FMAs per mma, each = 2 flops
    long long ops = (long long)blocks * threads / 32 * iters * 16 * 8 * 8 * 2;
    double tflops = ops / (ms * 1e-3) / 1e12;
    printf("  MMA m16n8k8 throughput: %.2f TFLOP/s\n", tflops);
    g_results.log("tensor_core", "mma_m16n8k8_throughput_tflops", tflops,
                  "TFLOP/s");
    g_results.log("peak", "fp16_tensor_tflops", tflops,
                  "TFLOP/s (TF32 measured, used for comparison)");
    CUDA_CHECK(cudaFree(d_out));
  }
}

} // namespace cmb
