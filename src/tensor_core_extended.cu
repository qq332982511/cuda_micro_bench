#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace cmb {

// Keep result alive without touching memory in the hot loop.
__device__ __noinline__ void sink_u32x2(uint32_t v0, uint32_t v1) {
  volatile uint32_t s[2] = {v0, v1};
  (void)s[0];
}
__device__ __noinline__ void sink_f32x4(float v0, float v1, float v2,
                                         float v3) {
  volatile float s[4] = {v0, v1, v2, v3};
  (void)s[0];
}
__device__ __noinline__ void sink_s32x4(int v0, int v1, int v2, int v3) {
  volatile int s[4] = {v0, v1, v2, v3};
  (void)s[0];
}

static __device__ __forceinline__ float half2_to_float(uint32_t v, int idx) {
  __half_raw h;
  h.x = (v >> (idx * 16)) & 0xffff;
  return __half2float(h);
}

// ------------------------------------------------------------------
// FP16 m16n8k16 (f16.f16.f16.f16) using packed .f16x2 operands
// ------------------------------------------------------------------
__global__ void kernel_hmma_f16_latency(unsigned int *cycles, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t a[4], b[2], c[2];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00; // 1.0, 1.0
  for (int i = 0; i < 2; ++i) b[i] = 0x3c003c00;
  for (int i = 0; i < 2; ++i) c[i] = 0;          // 0.0, 0.0

  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, "
        "{%2,%3,%4,%5}, "
        "{%6,%7}, "
        "{%8,%9};"
        : "=r"(c[0]), "=r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]));
  }
  CMB_CLOCK_STOP();
  sink_u32x2(c[0], c[1]);
  cycles[0] = t1 - t0;
#endif
}

__global__ void kernel_hmma_f16_throughput(float *out, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t a[4], b[2], c[2];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00;
  for (int i = 0; i < 2; ++i) b[i] = 0x3c003c00;
  for (int i = 0; i < 2; ++i) c[i] = 0;

  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, "
        "{%2,%3,%4,%5}, "
        "{%6,%7}, "
        "{%8,%9};"
        : "=r"(c[0]), "=r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] =
      half2_to_float(c[0], 0) + half2_to_float(c[0], 1) +
      half2_to_float(c[1], 0) + half2_to_float(c[1], 1);
#endif
}

// ------------------------------------------------------------------
// BF16 m16n8k16 (f32.bf16.bf16.f32) using packed .bf16x2 operands
// ------------------------------------------------------------------
__global__ void kernel_hmma_bf16_latency(unsigned int *cycles, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t a[4], b[2];
  float c[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 0x3f803f80; // 1.0, 1.0
  for (int i = 0; i < 2; ++i) b[i] = 0x3f803f80;
  for (int i = 0; i < 4; ++i) c[i] = 0.0f;

  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
  CMB_CLOCK_STOP();
  sink_f32x4(c[0], c[1], c[2], c[3]);
  cycles[0] = t1 - t0;
#endif
}

__global__ void kernel_hmma_bf16_throughput(float *out, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t a[4], b[2];
  float c[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 0x3f803f80;
  for (int i = 0; i < 2; ++i) b[i] = 0x3f803f80;
  for (int i = 0; i < 4; ++i) c[i] = 0.0f;

  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = c[0] + c[1] + c[2] + c[3];
#endif
}

// ------------------------------------------------------------------
// FP8 m16n8k32 (f32.e4m3.e4m3.f32)
// ------------------------------------------------------------------
__global__ void kernel_hmma_fp8_latency(unsigned int *cycles, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  int a[4], b[2];
  float c[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 0x01010101;
  for (int i = 0; i < 2; ++i) b[i] = 0x01010101;
  for (int i = 0; i < 4; ++i) c[i] = 0.0f;

  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
  CMB_CLOCK_STOP();
  sink_f32x4(c[0], c[1], c[2], c[3]);
  cycles[0] = t1 - t0;
#endif
}

__global__ void kernel_hmma_fp8_throughput(float *out, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  int a[4], b[2];
  float c[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 0x01010101;
  for (int i = 0; i < 2; ++i) b[i] = 0x01010101;
  for (int i = 0; i < 4; ++i) c[i] = 0.0f;

  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = c[0] + c[1] + c[2] + c[3];
#endif
}

// ------------------------------------------------------------------
// INT8 m16n8k32 (s32.s8.s8.s32)
// ------------------------------------------------------------------
__global__ void kernel_hmma_int8_latency(unsigned int *cycles, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  int a[4], b[2];
  int c[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 0x01010101;
  for (int i = 0; i < 2; ++i) b[i] = 0x01010101;
  for (int i = 0; i < 4; ++i) c[i] = 0;

  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
  }
  CMB_CLOCK_STOP();
  sink_s32x4(c[0], c[1], c[2], c[3]);
  cycles[0] = t1 - t0;
#endif
}

__global__ void kernel_hmma_int8_throughput(int *out, int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  int a[4], b[2];
  int c[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 0x01010101;
  for (int i = 0; i < 2; ++i) b[i] = 0x01010101;
  for (int i = 0; i < 4; ++i) c[i] = 0;

  for (int r = 0; r < iters; ++r) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = c[0] + c[1] + c[2] + c[3];
#endif
}

// ------------------------------------------------------------------
// Host driver helpers
// ------------------------------------------------------------------
static void measure_mma_latency(void (*kernel)(unsigned int *, int),
                                const char *label, int iters) {
  DeviceInfo info = queryDevice(0);
  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
  kernel<<<1, 32>>>(d_cycles, iters);
  CUDA_CHECK(cudaGetLastError());
  unsigned int h_cycles;
  CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  double cyc = (double)h_cycles / iters;
  double ns = cyc * 1e9 / info.clock_rate_khz;
  printf("  %-18s latency: %.2f cycles (%.2f ns)\n", label, cyc, ns);
  g_results.log("tensor_core_extended", std::string(label) + "_latency_cycles",
                cyc, "cycles");
  g_results.log("tensor_core_extended", std::string(label) + "_latency_ns", ns,
                "ns");
  CUDA_CHECK(cudaFree(d_cycles));
}

static void measure_mma_throughput_fp(void (*kernel)(float *, int),
                                      const char *label, int iters,
                                      long long ops_per_mma) {
  DeviceInfo info = queryDevice(0);
  int threads = 32;
  int blocks = info.sm_count * info.max_warps_per_sm;
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

  long long ops = (long long)blocks * threads / 32 * iters * ops_per_mma;
  double tflops = ops / (ms * 1e-3) / 1e12;
  printf("  %-18s throughput: %.2f TFLOP/s\n", label, tflops);
  g_results.log("tensor_core_extended", std::string(label) + "_throughput_TFs",
                tflops, "TFLOP/s");
  CUDA_CHECK(cudaFree(d_out));
}

static void measure_mma_throughput_int(void (*kernel)(int *, int),
                                       const char *label, int iters,
                                       long long ops_per_mma) {
  DeviceInfo info = queryDevice(0);
  int threads = 32;
  int blocks = info.sm_count * info.max_warps_per_sm;
  int *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(int)));

  kernel<<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK(cudaGetLastError());

  GpuTimer timer;
  timer.start();
  kernel<<<blocks, threads>>>(d_out, iters);
  timer.stop();
  float ms = timer.elapsedMs();
  CUDA_CHECK(cudaGetLastError());

  long long ops = (long long)blocks * threads / 32 * iters * ops_per_mma;
  double tops = ops / (ms * 1e-3) / 1e12;
  printf("  %-18s throughput: %.2f TOP/s\n", label, tops);
  g_results.log("tensor_core_extended", std::string(label) + "_throughput_Tops",
                tops, "TOP/s");
  CUDA_CHECK(cudaFree(d_out));
}

void benchmark_tensor_core_extended() {
  DeviceInfo info = queryDevice(0);
  printf("=== Tensor Core Extended (MMA) ===\n");

  int cc = info.cc_major * 10 + info.cc_minor;
  if (cc < 80) {
    printf("  Extended tensor core benchmarks require compute capability >= "
           "8.0. Skipping.\n");
    return;
  }

  int iters = 1 << 12;

  // FP16 m16n8k16: 16*8*16 = 2048 FMAs = 4096 flops
  measure_mma_latency(kernel_hmma_f16_latency, "fp16_m16n8k16", iters);
  measure_mma_throughput_fp(kernel_hmma_f16_throughput, "fp16_m16n8k16", iters,
                            16LL * 8 * 16 * 2);

  // BF16 m16n8k16: same flop count
  measure_mma_latency(kernel_hmma_bf16_latency, "bf16_m16n8k16", iters);
  measure_mma_throughput_fp(kernel_hmma_bf16_throughput, "bf16_m16n8k16", iters,
                            16LL * 8 * 16 * 2);

  if (cc >= 89) {
    // FP8 m16n8k32: 16*8*32 = 4096 FMAs = 8192 flops
    measure_mma_latency(kernel_hmma_fp8_latency, "fp8_m16n8k32", iters);
    measure_mma_throughput_fp(kernel_hmma_fp8_throughput, "fp8_m16n8k32", iters,
                              16LL * 8 * 32 * 2);

    // INT8 m16n8k32: 16*8*32 = 4096 MACs = 4096 ops
    measure_mma_latency(kernel_hmma_int8_latency, "int8_m16n8k32", iters);
    measure_mma_throughput_int(kernel_hmma_int8_throughput, "int8_m16n8k32",
                               iters, 16LL * 8 * 32);
  } else {
    printf("  FP8/INT8 m16n8k32 require compute capability >= 8.9. Skipping.\n");
  }
}

} // namespace cmb
