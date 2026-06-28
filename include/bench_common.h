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

// Helper to flush caches
inline void flushL2() {
  int dev;
  cudaGetDevice(&dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  size_t size = prop.l2CacheSize * 2;
  char *buf;
  cudaMalloc(&buf, size);
  cudaMemset(buf, 0, size);
  cudaFree(buf);
}

} // namespace cmb
