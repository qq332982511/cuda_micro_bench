#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

// Latency of a dependent shfl.sync.idx chain.
__global__ void kernel_shfl_latency(unsigned int *cycles, int iters) {
  int lane = (threadIdx.x + 1) & 31;
  int src = threadIdx.x;
  unsigned int mask = 0xffffffff;

  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    int r;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(r)
                 : "r"(src), "r"(lane), "r"(mask));
    src = r;
  }
  CMB_CLOCK_STOP();
  if (src == 0xdeadbeef)
    t1 = clockVal();
  cycles[0] = t1 - t0;
}

// Throughput: independent shfl chains (ILP=4) spread over many warps.
__global__ void kernel_shfl_throughput(int *out, int iters) {
  int lane = (threadIdx.x + 1) & 31;
  unsigned int mask = 0xffffffff;
  int a0 = threadIdx.x;
  int a1 = threadIdx.x + 1;
  int a2 = threadIdx.x + 2;
  int a3 = threadIdx.x + 3;

  for (int i = 0; i < iters; ++i) {
    int r0, r1, r2, r3;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(r0)
                 : "r"(a0), "r"(lane), "r"(mask));
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(r1)
                 : "r"(a1), "r"(lane), "r"(mask));
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(r2)
                 : "r"(a2), "r"(lane), "r"(mask));
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(r3)
                 : "r"(a3), "r"(lane), "r"(mask));
    a0 = r0;
    a1 = r1;
    a2 = r2;
    a3 = r3;
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

void benchmark_warp_shuffle() {
  DeviceInfo info = queryDevice(0);
  printf("=== Warp Shuffle ===\n");

  // Latency
  {
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    int iters = 1 << 14;
    kernel_shfl_latency<<<1, 32>>>(d_cycles, iters);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h_cycles;
    CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double cyc = (double)h_cycles / iters;
    double ns = cyc * 1e9 / info.clock_rate_khz;
    printf("  shfl.sync.idx latency: %.2f cycles (%.2f ns)\n", cyc, ns);
    g_results.log("warp_shuffle", "shfl_idx_latency_cycles", cyc, "cycles");
    g_results.log("warp_shuffle", "shfl_idx_latency_ns", ns, "ns");
    CUDA_CHECK(cudaFree(d_cycles));
  }

  // Throughput
  {
    int iters = 1 << 14;
    int threads = 32;
    int blocks = info.sm_count * info.max_warps_per_sm;
    int *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(int)));

    kernel_shfl_throughput<<<blocks, threads>>>(d_out, iters);
    CUDA_CHECK(cudaGetLastError());

    GpuTimer timer;
    timer.start();
    kernel_shfl_throughput<<<blocks, threads>>>(d_out, iters);
    timer.stop();
    float ms = timer.elapsedMs();
    CUDA_CHECK(cudaGetLastError());

    long long insts = (long long)blocks * threads * iters * 4;
    double ginst = insts / (ms * 1e-3) / 1e9;
    printf("  shfl.sync.idx throughput: %.2f Ginst/s\n", ginst);
    g_results.log("warp_shuffle", "shfl_idx_throughput_Ginst_s", ginst, "Ginst/s");
    CUDA_CHECK(cudaFree(d_out));
  }
}

} // namespace cmb
