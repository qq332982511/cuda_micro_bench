#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

// Stream bandwidth: each thread reads/writes contiguous elements
__global__ void kernel_stream_copy(const float *src, float *dst, int n,
                                   int iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    float v = src[i];
    for (int r = 0; r < iters; ++r) {
      v = v + 1.0f;
    }
    dst[i] = v;
  }
}

__global__ void kernel_latency_chase(const int *indices, int n, int iters,
                                     unsigned int *out) {
  int idx = 0;
  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    int v;
    // Volatile PTX load keeps the chase dependency chain; the final index is
    // written to out[1] so the loop cannot be eliminated.
    asm volatile("ld.global.b32 %0, [%1];" : "=r"(v) : "l"(indices + idx));
    idx = v;
  }
  CMB_CLOCK_STOP();
  out[0] = t1 - t0;
  out[1] = (unsigned int)idx;
}

void benchmark_global_memory() {
  DeviceInfo info = queryDevice(0);
  printf("=== Global Memory Benchmarks ===\n");

  // 1. Stream bandwidth with increasing working sets.  Use the largest size
  // that fits as the reported value because it is least sensitive to launch
  // and timer overhead; smaller sizes are printed for sanity checking.
  const size_t sizes[] = {16 << 20, 64 << 20, 128 << 20, 256 << 20};
  const int inner_iters = 1;
  double reported_bw = 0.0;
  size_t reported_bytes = 0;
  for (size_t bytes : sizes) {
    if (bytes > info.total_global_mem / 4)
      continue;
    int n = bytes / sizeof(float);
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMemset(d_a, 0, bytes));

    int threads = 256;
    int blocks = info.sm_count * 8;

    // Warm-up launch and sync.
    kernel_stream_copy<<<blocks, threads>>>(d_a, d_b, n, inner_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    kernel_stream_copy<<<blocks, threads>>>(d_a, d_b, n, inner_iters);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.elapsedMs();

    // Each element is read once and written once.
    double total_bytes = bytes * 2.0;
    double bw = bandwidthGiB(total_bytes, ms * 1e6);
    printf("  Copy %6zu MB -> %.3f ms, BW: %.2f GiB/s\n",
           bytes / (1024 * 1024), ms, bw);
    if (bytes > reported_bytes) {
      reported_bw = bw;
      reported_bytes = bytes;
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
  }
  if (reported_bytes > 0) {
    g_results.log("memory_dram", "copy_max_bw_gibps", reported_bw, "GiB/s");
    // Also report in GB/s under the canonical memory section for the HTML report.
    g_results.log("memory", "dram_read_bw_gbps", reported_bw * 1.073741824, "GB/s");
  }

  // 2. Pointer-chase latency
  const int chase_size = 1 << 20; // 4 MB
  int *d_indices;
  CUDA_CHECK(cudaMalloc(&d_indices, chase_size * sizeof(int)));
  // Build a random-ish linked list in array
  std::vector<int> h_indices(chase_size);
  for (int i = 0; i < chase_size; ++i)
    h_indices[i] = (i + 64) % chase_size;
  CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), chase_size * sizeof(int),
                        cudaMemcpyHostToDevice));

  unsigned int *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(unsigned int)));
  kernel_latency_chase<<<1, 1>>>(d_indices, chase_size, 1 << 20, d_out);
  CUDA_CHECK(cudaGetLastError());
  unsigned int h_out[2];
  CUDA_CHECK(cudaMemcpy(h_out, d_out, 2 * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  double lat_cyc = (double)h_out[0] / (1 << 20);
  double lat_ns = lat_cyc * 1e9 / info.clock_rate_khz;
  printf("  Pointer-chase latency: %.2f cycles (%.2f ns)\n", lat_cyc, lat_ns);
  g_results.log("memory_dram", "pointer_chase_latency_ns", lat_ns, "ns");
  g_results.log("memory", "dram_latency_cyc", lat_cyc, "cycles");

  CUDA_CHECK(cudaFree(d_indices));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace cmb
