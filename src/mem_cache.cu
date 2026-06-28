#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

namespace cmb {

// Cache size probe: access arrays of increasing size, measure avg latency
__global__ void kernel_cache_probe(const float *data, int n, int iters,
                                   unsigned int *cycles) {
  int tid = threadIdx.x;
  int idx = tid % n;
  float acc = 0.0f;
  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    acc += data[idx];
    idx = (idx + 128) % n; // stride to stay in cache lines
  }
  CMB_CLOCK_STOP();
  if (acc < 0.0f)
    cycles[tid] = (unsigned int)(acc);
  else
    cycles[tid] = t1 - t0;
}

// L1 capacity probe via pointer chase through increasing working sets.
__global__ void kernel_l1_chase(const int *indices, int n, int iters,
                                unsigned int *cycles) {
  int idx = 0;
  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    int v;
    // Volatile PTX load prevents the compiler from collapsing the dependency
    // chain; the final idx is stored alongside the timer so the loop cannot be
    // DCE'd, while cycles[0] remains a pure clock difference.
    asm volatile("ld.global.b32 %0, [%1];" : "=r"(v) : "l"(indices + idx));
    idx = v;
  }
  CMB_CLOCK_STOP();
  cycles[0] = t1 - t0;
  cycles[1] = (unsigned int)idx;
}

// L1/L2 cache line size probe: stride access
__global__ void kernel_stride_access(const float *data, int n, int stride,
                                     int iters, unsigned int *cycles) {
  int tid = threadIdx.x;
  int idx = (tid * stride) % n;
  float acc = 0.0f;
  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    acc += data[idx];
    idx = (idx + stride) % n;
  }
  CMB_CLOCK_STOP();
  if (acc < 0.0f)
    cycles[tid] = (unsigned int)(acc);
  else
    cycles[tid] = t1 - t0;
}

void benchmark_cache() {
  DeviceInfo info = queryDevice(0);
  printf("=== Cache Benchmarks ===\n");

  // 1. Cache size / latency hierarchy probe
  const int min_kb = 4;
  const int max_kb = 64 * 1024; // up to 64 MB
  std::vector<unsigned int> h_cycles(256);
  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, 256 * sizeof(unsigned int)));

  printf("  Cache size/latency probe (stride=128, one warp):\n");
  for (int kb = min_kb; kb <= max_kb; kb *= 2) {
    size_t bytes = (size_t)kb * 1024;
    if (bytes > info.total_global_mem / 8)
      break;
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));

    kernel_cache_probe<<<1, 32>>>(d_data, bytes / sizeof(float), 1 << 14,
                                  d_cycles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_cycles.data(), d_cycles, 32 * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    unsigned long long sum = 0;
    for (int i = 0; i < 32; ++i)
      sum += h_cycles[i];
    double cyc = (double)sum / 32.0 / (1 << 14);
    double ns = cyc * 1e9 / info.clock_rate_khz;
    printf("    %6d KB -> %.2f cycles (%.2f ns)\n", kb, cyc, ns);

    if (kb == 4 || kb == 256 || kb == 4096 || kb == 32768 || kb == 65536) {
      char key[64];
      snprintf(key, sizeof(key), "latency_%d_kb_cycles", kb);
      g_results.log("memory_cache", key, cyc, "cycles");
      char ns_key[64];
      snprintf(ns_key, sizeof(ns_key), "latency_%d_kb_ns", kb);
      g_results.log("memory_cache", ns_key, ns, "ns");
    }
    if (kb == 4) {
      g_results.log("memory", "l1_latency_cyc", cyc, "cycles");
    }
    if (kb == info.l2_cache_size / 1024) {
      g_results.log("memory", "l2_latency_cyc", cyc, "cycles");
    }

    CUDA_CHECK(cudaFree(d_data));
  }

  // 2. Cache line size probe
  printf("  Stride latency probe (8 MB array):\n");
  size_t bytes = 8ULL * 1024 * 1024;
  float *d_data;
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMemset(d_data, 0, bytes));
  for (int stride_bytes = 4; stride_bytes <= 512; stride_bytes *= 2) {
    kernel_stride_access<<<1, 32>>>(d_data, bytes / sizeof(float),
                                    stride_bytes / sizeof(float), 1 << 14,
                                    d_cycles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_cycles.data(), d_cycles, 32 * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    unsigned long long sum = 0;
    for (int i = 0; i < 32; ++i)
      sum += h_cycles[i];
    double cyc = (double)sum / 32.0 / (1 << 14);
    printf("    stride %4d B -> %.2f cycles\n", stride_bytes, cyc);
    if (stride_bytes == 4 || stride_bytes == 16 || stride_bytes == 64 ||
        stride_bytes == 128 || stride_bytes == 256 || stride_bytes == 512) {
      char key[64];
      snprintf(key, sizeof(key), "stride_%d_bytes_cycles", stride_bytes);
      g_results.log("memory_cache", key, cyc, "cycles");
    }
  }
  CUDA_CHECK(cudaFree(d_data));

  // 3. L1 capacity probe: pointer chase through working sets 8 KB .. 256 KB
  printf("  L1 capacity probe (pointer chase):\n");
  for (int kb = 8; kb <= 256; kb *= 2) {
    int n = kb * 1024 / sizeof(int);
    int *d_idx;
    CUDA_CHECK(cudaMalloc(&d_idx, n * sizeof(int)));
    std::vector<int> h_idx(n);
    for (int i = 0; i < n; ++i)
      h_idx[i] = (i + 64) % n;
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));

    kernel_l1_chase<<<1, 1>>>(d_idx, n, 1 << 16, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h_c[2];
    CUDA_CHECK(cudaMemcpy(h_c, d_cycles, 2 * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double cyc = (double)h_c[0] / (1 << 16);
    printf("    %4d KB -> %.2f cycles/access\n", kb, cyc);
    char key[64];
    snprintf(key, sizeof(key), "l1_chase_%d_kb_cycles", kb);
    g_results.log("memory_cache", key, cyc, "cycles");
    if (kb == 64) {
      g_results.log("memory", "l1_latency_cyc", cyc, "cycles");
    }
    CUDA_CHECK(cudaFree(d_idx));
  }

  CUDA_CHECK(cudaFree(d_cycles));
}

} // namespace cmb
