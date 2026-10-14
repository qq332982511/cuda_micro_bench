#pragma once

#include "utils.h"
#include "timer.h"
#include <cuda_runtime.h>

namespace cmb {

// Common launch config
constexpr int WARP_SIZE = 32;
constexpr int DEFAULT_ITERS = 1024;
constexpr int DEFAULT_WARPS = 32;

inline int divUp(int a, int b) { return (a + b - 1) / b; }

__global__ void kernel_init_stream(unsigned int *data, int n);

// Helper to flush caches
inline void flushL2() {
  int dev;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  size_t size = prop.l2CacheSize * 2;
  char *buf;
  CUDA_CHECK(cudaMalloc(&buf, size));
  CUDA_CHECK(cudaMemset(buf, 0, size));
  CUDA_CHECK(cudaFree(buf));
}

} // namespace cmb
