#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

namespace cmb {

// Host timer using CUDA events
class GpuTimer {
public:
  GpuTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start() { cudaEventRecord(start_, 0); }
  void stop() { cudaEventRecord(stop_, 0); }
  float elapsedMs() {
    cudaEventSynchronize(stop_);
    float ms;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

private:
  cudaEvent_t start_, stop_;
};

// Device clock access via PTX
__device__ __forceinline__ unsigned int clockVal() {
  unsigned int ret;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(ret));
  return ret;
}

__device__ __forceinline__ unsigned long long clock64Val() {
  unsigned long long ret;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(ret));
  return ret;
}

// PTX-based timer kernel helpers
#define CMB_CLOCK_START() unsigned int t0 = clockVal();
#define CMB_CLOCK_STOP() unsigned int t1 = clockVal();

} // namespace cmb
