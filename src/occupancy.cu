#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

// Dummy kernel with configurable shared memory and register usage
__global__ void __launch_bounds__(1024, 1)
kernel_occupancy(float *in, float *out, int n, int iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    acc += in[i];
  }
  for (int r = 0; r < iters; ++r) {
    acc = fma(acc, 1.0001f, 1.0f);
  }
  if (idx < n)
    out[idx] = acc;
}

void benchmark_occupancy() {
  DeviceInfo info = queryDevice(0);
  printf("=== Occupancy / Launch Limits ===\n");

  int block_sizes[] = {64, 128, 256, 512, 1024};
  for (int bs : block_sizes) {
    int min_grid, best_block;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block,
                                       kernel_occupancy, 0, 0));
    int max_blocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel_occupancy,
                                                  bs, 0));
    printf("  Block size %4d -> max blocks/SM: %d\n", bs, max_blocks);
    char key[64];
    snprintf(key, sizeof(key), "max_blocks_per_sm_block_%d", bs);
    g_results.log("occupancy", key, max_blocks, "blocks");
  }

  // Register pressure effect: launch with many threads and measure
  size_t bytes = 64ULL * 1024 * 1024;
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemset(d_in, 0, bytes));

  int n = bytes / sizeof(float);
  int threads = 256;
  int blocks = info.sm_count * 4;

  GpuTimer timer;
  timer.start();
  kernel_occupancy<<<blocks, threads>>>(d_in, d_out, n, 1 << 10);
  timer.stop();
  CUDA_CHECK(cudaGetLastError());
  float ms = timer.elapsedMs();
  printf("  Occupancy kernel %d blocks x %d threads: %.3f ms\n", blocks, threads,
         ms);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

} // namespace cmb
