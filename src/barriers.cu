#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

extern __shared__ char smem_raw[];

// __syncthreads() latency measured by one warp in a block.
__global__ void kernel_syncthreads_1warp(unsigned int *cycles, int iters) {
  int sink = threadIdx.x;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    __syncthreads();
    sink += i;
  }
  CMB_CLOCK_STOP();
  if (sink == 0xdeadbeef)
    t1 = clockVal();
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
}

// __syncthreads() latency measured by a full block.
__global__ void kernel_syncthreads_full(unsigned int *cycles, int iters) {
  int sink = threadIdx.x;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    __syncthreads();
    sink += i;
  }
  CMB_CLOCK_STOP();
  if (sink == 0xdeadbeef)
    t1 = clockVal();
  cycles[threadIdx.x] = t1 - t0;
}

// Named barrier (bar.sync) latency measured by one warp.
__global__ void kernel_named_barrier(unsigned int *cycles, int iters) {
  int sink = threadIdx.x;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    asm volatile("bar.sync 1;" ::: "memory");
    sink += i;
  }
  CMB_CLOCK_STOP();
  if (sink == 0xdeadbeef)
    t1 = clockVal();
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
}

// mbarrier arrive/wait latency (single thread, no async op, tx_count=0).
__global__ void kernel_mbarrier_latency(unsigned int *cycles, int iters) {
  __shared__ alignas(8) uint64_t mbar;
  uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(mbar_addr), "r"(1));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    CMB_CLOCK_START();
    for (int i = 0; i < iters; ++i) {
      uint64_t token __attribute__((unused));
      asm volatile("mbarrier.arrive.release.cta.shared.b64 %0, [%1], %2;"
                   : "=l"(token)
                   : "r"(mbar_addr), "r"(1));
      unsigned int parity = i & 1;
      unsigned int ready = 0;
      do {
        asm volatile(
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared.b64 p, [%1], %2;\n\t"
            "selp.b32 %0, 1, 0, p;"
            : "=r"(ready)
            : "r"(mbar_addr), "r"(parity));
      } while (!ready);
    }
    CMB_CLOCK_STOP();
    if (threadIdx.x == 0) cycles[0] = t1 - t0;
  }
}

// Converged warp-sync loop. ptxas eliminates bar.warp.sync on this target;
// the observed value is loop cost, not WARPSYNC instruction latency.
__global__ void kernel_syncwarp(unsigned int *cycles, int iters) {
  int sink = threadIdx.x;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
    sink += i;
  }
  CMB_CLOCK_STOP();
  if (sink == 0xdeadbeef)
    t1 = clockVal();
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
}

#define FENCE_KERNEL(name, space)                                              \
  __global__ void name(unsigned int *cycles, int iters) {                      \
    int sink = threadIdx.x;                                                    \
    CMB_CLOCK_START();                                                         \
    for (int i = 0; i < iters; ++i) {                                          \
      asm volatile("fence.acq_rel." #space ";" ::: "memory");                  \
      sink += i;                                                               \
    }                                                                          \
    CMB_CLOCK_STOP();                                                          \
    if (sink == 0xdeadbeef)                                                    \
      t1 = clockVal();                                                         \
    if (threadIdx.x == 0) cycles[0] = t1 - t0;                                                       \
  }

FENCE_KERNEL(kernel_fence_cta, cta)
FENCE_KERNEL(kernel_fence_gpu, gpu)
FENCE_KERNEL(kernel_fence_sys, sys)

static double measure_barrier(void (*kernel)(unsigned int *, int),
                              int block_size, int iters,
                              const char *label) {
  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, block_size * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(d_cycles, 0, block_size * sizeof(unsigned int)));

  kernel<<<1, block_size>>>(d_cycles, iters);
  CUDA_CHECK(cudaGetLastError());

  unsigned int h_cycles[1024];
  CUDA_CHECK(cudaMemcpy(h_cycles, d_cycles, block_size * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  // Use thread 0's measurement.
  double cyc = (double)h_cycles[0] / iters;
  CUDA_CHECK(cudaFree(d_cycles));
  return cyc;
}

void benchmark_barriers() {
  DeviceInfo info = queryDevice(0);
  printf("=== Barriers ===\n");
  int iters = 1 << 12;

  double cyc;
  double ns;

  // 1 warp only
  cyc = measure_barrier(kernel_syncthreads_1warp, 32, iters, "syncthreads_1warp");
  ns = cyclesToNominalNs(cyc, info.clock_rate_khz);
  printf("  __syncthreads() latency (1 warp):  %.2f cycles (%.2f nominal ns)\n", cyc, ns);
  g_results.log("barriers", "syncthreads_1warp_cycles", cyc, "cycles");
  g_results.log("barriers", "syncthreads_1warp_ns", ns, "nominal ns");

  // Full block
  cyc = measure_barrier(kernel_syncthreads_full, 256, iters, "syncthreads_full");
  ns = cyclesToNominalNs(cyc, info.clock_rate_khz);
  printf("  __syncthreads() latency (256 threads): %.2f cycles (%.2f nominal ns)\n", cyc, ns);
  g_results.log("barriers", "syncthreads_full_cycles", cyc, "cycles");
  g_results.log("barriers", "syncthreads_full_ns", ns, "nominal ns");

  // Named barrier
  cyc = measure_barrier(kernel_named_barrier, 32, iters, "named_barrier");
  ns = cyclesToNominalNs(cyc, info.clock_rate_khz);
  printf("  bar.sync 1 latency:                %.2f cycles (%.2f nominal ns)\n", cyc, ns);
  g_results.log("barriers", "named_barrier_cycles", cyc, "cycles");
  g_results.log("barriers", "named_barrier_ns", ns, "nominal ns");

  // mbarrier arrive/wait
  cyc = measure_barrier(kernel_mbarrier_latency, 32, iters, "mbarrier");
  ns = cyclesToNominalNs(cyc, info.clock_rate_khz);
  printf("  mbarrier arrive+wait latency:      %.2f cycles (%.2f nominal ns)\n", cyc, ns);
  g_results.log("barriers", "mbarrier_cycles", cyc, "cycles");
  g_results.log("barriers", "mbarrier_ns", ns, "nominal ns");

  // __syncwarp
  cyc = measure_barrier(kernel_syncwarp, 32, iters, "syncwarp");
  ns = cyclesToNominalNs(cyc, info.clock_rate_khz);
  printf("  Converged syncwarp loop cost:              %.2f cycles (%.2f nominal ns)\n", cyc, ns);
  g_results.log("barriers", "converged_syncwarp_loop_cycles", cyc, "cycles");
  g_results.log("barriers", "syncwarp_ns", ns, "nominal ns");

  // Fences
  const char *fence_names[] = {"cta", "gpu", "sys"};
  void (*fence_kernels[])(unsigned int *, int) = {kernel_fence_cta,
                                                  kernel_fence_gpu,
                                                  kernel_fence_sys};
  for (int f = 0; f < 3; ++f) {
    cyc = measure_barrier(fence_kernels[f], 1, iters, fence_names[f]);
    ns = cyclesToNominalNs(cyc, info.clock_rate_khz);
    printf("  fence.acq_rel.%s latency:         %.2f cycles (%.2f nominal ns)\n",
           fence_names[f], cyc, ns);
    char key[64];
    snprintf(key, sizeof(key), "fence_%s_cycles", fence_names[f]);
    g_results.log("barriers", key, cyc, "cycles");
  }
}

} // namespace cmb
