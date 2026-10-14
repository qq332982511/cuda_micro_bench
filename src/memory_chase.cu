#include "memory_chase.h"

namespace cmb {
// One randomized cycle visits EVERY 128-byte slot exactly once. Cache policy
// is explicit: .ca for L1 capacity, .cg to bypass L1 for L2 / DRAM probes.
template <bool BYPASS_L1>
__device__ __forceinline__ void randomChase(const unsigned int *indices, int warmup,
                                   int iters, unsigned long long *out) {
  unsigned int idx = 0;
  for (int r = 0; r < warmup; ++r) {
    if constexpr (BYPASS_L1)
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(idx) : "l"(indices + idx) : "memory");
    else
      asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(idx) : "l"(indices + idx) : "memory");
  }
  unsigned long long t0 = clockAfter64(idx);
#pragma unroll 32
  for (int r = 0; r < iters; ++r) {
    if constexpr (BYPASS_L1)
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(idx) : "l"(indices + idx) : "memory");
    else
      asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(idx) : "l"(indices + idx) : "memory");
  }
  unsigned long long t1 = clockAfter64(idx);
  out[0] = t1 - t0;
  out[1] = idx;
}

__global__ void kernel_random_chase_ca(const unsigned int *indices, int warmup, int iters, unsigned long long *out) {
  randomChase<false>(indices, warmup, iters, out);
}
__global__ void kernel_random_chase_cg(const unsigned int *indices, int warmup, int iters, unsigned long long *out) {
  randomChase<true>(indices, warmup, iters, out);
}
} // namespace cmb
