#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

namespace cmb {

// Wall-clock kernel: each active warp runs an independent FFMA dependency chain.
// A single thread records the block-level clock around a __syncthreads() so the
// measurement reflects scheduler interleaving of all warps.
__global__ void kernel_warp_scaling(int warps, int iters,
                                    unsigned int *out_cycles) {
  extern __shared__ float smem[]; // one float per warp
  int tid = threadIdx.x;
  int warp = tid / 32;

  __shared__ unsigned int s_t0, s_t1;

  __syncthreads();
  if (tid == 0)
    s_t0 = clockVal();
  __syncthreads();

  float a = 1.0f + warp * 0.001f;
  const float b = 1.0001f;
  // Each warp performs its own dep-chain; the scheduler must interleave them.
  if (warp < warps) {
    for (int i = 0; i < iters; ++i) {
      float r;
      asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                   : "=f"(r)
                   : "f"(a), "f"(b), "f"(a));
      a = r;
    }
  }
  smem[warp] = a;

  __syncthreads();
  if (tid == 0)
    s_t1 = clockVal();
  __syncthreads();

  if (tid == 0)
    out_cycles[0] = s_t1 - s_t0;
}

// Minimal warp-to-SMSP consistency probe: compare cycles/FMA with 1 warp vs
// 4 warps.  If 4 warps run roughly 4x faster (per FMA) than 1 warp, the
// scheduler is distributing warps across the 4 SMSPs in a round-robin fashion.
__global__ void kernel_warp_to_smsp_probe(int warps, int iters,
                                          unsigned int *out) {
  int tid = threadIdx.x;
  int warp = tid / 32;
  __shared__ unsigned int s_t0, s_t1;
  __shared__ float sink[4];
  float a = 1.0f + warp * 0.001f;
  const float b = 1.0001f;
  __syncthreads();
  if (tid == 0)
    s_t0 = clockVal();
  __syncthreads();
  if (warp < warps) {
    for (int i = 0; i < iters; ++i) {
      float r;
      asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                   : "=f"(r) : "f"(a), "f"(b), "f"(a));
      a = r;
    }
    sink[warp] = a;
  }
  __syncthreads();
  if (tid == 0)
    s_t1 = clockVal();
  __syncthreads();
  if (tid == 0)
    out[0] = s_t1 - s_t0;
  if (sink[0] == 0xdeadbeef)
    out[0] = 0;
}

void benchmark_topology_extended() {
  DeviceInfo info = queryDevice(0);
  printf("=== Topology / SMSP Probes ===\n");

  int max_warps = info.max_warps_per_sm;
  int smsps = 4; // SM_120 has 4 SMSPs per SM (documented; verified by scaling)
  int warps_per_smsp = max_warps / smsps;
  g_results.log("topology", "smsps_per_sm", smsps, "");
  g_results.log("topology", "warps_per_smsp", warps_per_smsp, "");

  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));

  // Warp-to-SMSP round-robin check.
  {
    int iters = 1 << 14;
    int threads = smsps * 32;
    kernel_warp_to_smsp_probe<<<1, threads>>>(1, iters, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h1;
    CUDA_CHECK(cudaMemcpy(&h1, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double cyc1 = (double)h1 / iters;

    kernel_warp_to_smsp_probe<<<1, threads>>>(smsps, iters, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h4;
    CUDA_CHECK(cudaMemcpy(&h4, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double cyc4 = (double)h4 / iters / smsps;

    double ratio = cyc1 / (cyc4 > 0 ? cyc4 : 1e-9);
    bool consistent = ratio >= 3.5;
    printf("  Warp-to-SMSP round-robin check: 1-warp %.2f cyc/FMA, 4-warps %.2f cyc/FMA, ratio %.2f (%s)\n",
           cyc1, cyc4, ratio, consistent ? "consistent" : "inconclusive");
    g_results.log("topology", "warp_to_smsp_round_robin_ratio", ratio, "");
    g_results.log("topology", "warp_to_smsp_consistent", consistent ? 1.0 : 0.0,
                  "bool");
  }

  const int iters = 1 << 14;
  // A single block is limited to maxThreadsPerBlock threads, so we can only
  // fit that many warps in one block.  Cap the sweep accordingly.
  int max_warps_per_block = info.max_threads_per_block / 32;
  int sweep_warps = std::min(max_warps, max_warps_per_block);
  std::vector<double> cyc_per_fma(sweep_warps + 1, 0.0);
  double baseline = 1e9;
  int full_latency_hiding_warps = 0;

  for (int w = 1; w <= sweep_warps; ++w) {
    int threads = w * 32;
    int smem_bytes = (max_warps + 1) * sizeof(float);
    kernel_warp_scaling<<<1, threads, smem_bytes>>>(w, iters, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h;
    CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    // Total FMAs issued = w * iters (one dep-chain per warp).
    double cyc = (double)h / (double)iters;
    cyc_per_fma[w] = cyc;
    if (cyc < baseline)
      baseline = cyc;
  }

  // Find the first warp count where cycles/FMA rises > 5% above the minimum.
  // On SM_120 this occurs when any SMSP gets more than 4 warps.
  for (int w = 1; w <= max_warps; ++w) {
    if (cyc_per_fma[w] > baseline * 1.05) {
      full_latency_hiding_warps = w - 1;
      break;
    }
  }
  if (full_latency_hiding_warps == 0)
    full_latency_hiding_warps = max_warps;

  printf("  SMSPs / SM            : %d\n", smsps);
  printf("  Warps / SMSP (max)    : %d\n", warps_per_smsp);
  printf("  Full latency hiding   : %d warps/SM (%.2f cyc/FMA baseline)\n",
         full_latency_hiding_warps, baseline);

  // Log a sparse staircase for the report.
  auto log_pt = [&](int w) {
    char key[64];
    snprintf(key, sizeof(key), "warp_scaling_w%d_cyc_per_fma", w);
    g_results.log("topology", key, cyc_per_fma[w], "cycles");
    printf("    W=%2d -> %.2f cyc/FMA\n", w, cyc_per_fma[w]);
  };
  for (int w : {1, 4, 5, 8, 12, 16, 20, 24, 32, 40, 48}) {
    if (w <= sweep_warps)
      log_pt(w);
  }

  g_results.log("topology", "full_latency_hiding_warps", full_latency_hiding_warps,
                "warps");
  g_results.log("topology", "warp_scaling_baseline_cyc_per_fma", baseline, "cycles");

  CUDA_CHECK(cudaFree(d_cycles));
}

} // namespace cmb
