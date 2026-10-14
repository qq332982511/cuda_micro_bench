#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "utils.h"

namespace cmb {

// Host timer using CUDA events
class GpuTimer {
public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    CUDA_CHECK(cudaEventDestroy(start_));
    CUDA_CHECK(cudaEventDestroy(stop_));
  }
  void start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
  void stop() { CUDA_CHECK(cudaEventRecord(stop_, 0)); }
  float elapsedMs() {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
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

// Both clock reads are predicated on the final chain result. Unlike a C++
// condition with an impossible sentinel, this survives ptxas optimization.
// Check the generated SASS when changing this helper.
__device__ __forceinline__ unsigned int clockAfter(unsigned int sink) {
  unsigned int ret;
  asm volatile("{ .reg .pred p;\n\t"
               "setp.eq.u32 p, %1, 0;\n\t"
               "@p mov.u32 %0, %%clock;\n\t"
               "@!p mov.u32 %0, %%clock; }"
               : "=r"(ret) : "r"(sink) : "memory");
  return ret;
}
__device__ __forceinline__ unsigned long long clockAfter64(unsigned int sink) {
  unsigned long long ret;
  asm volatile("{ .reg .pred p;\n\t"
               "setp.eq.u32 p, %1, 0;\n\t"
               "@p mov.u64 %0, %%clock64;\n\t"
               "@!p mov.u64 %0, %%clock64; }"
               : "=l"(ret) : "r"(sink) : "memory");
  return ret;
}
__device__ __forceinline__ unsigned int clockAfter(int sink) {
  return clockAfter(static_cast<unsigned int>(sink));
}
__device__ __forceinline__ unsigned int clockAfter(float sink) {
  return clockAfter(__float_as_uint(sink));
}
__device__ __forceinline__ unsigned int clockAfter(unsigned long long sink) {
  return clockAfter(static_cast<unsigned int>(sink) ^
                    static_cast<unsigned int>(sink >> 32));
}
__device__ __forceinline__ unsigned int clockAfter(double sink) {
  return clockAfter(static_cast<unsigned long long>(__double_as_longlong(sink)));
}

#define CMB_DEPENDENCY_GUARD(v) t1 = clockAfter(v)

// Attribute clocks are nominal, not a measurement of clocks during the kernel.
inline double cyclesToNominalNs(double cycles, int clock_khz) {
  return cycles * 1e6 / clock_khz;
}

// PTX-based timer kernel helpers
#define CMB_CLOCK_START() unsigned int t0 = clockVal();
#define CMB_CLOCK_STOP() unsigned int t1 = clockVal();

} // namespace cmb
