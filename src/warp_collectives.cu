#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

namespace cmb {

static const unsigned int FULL_MASK = 0xffffffffU;

// Latency of vote.ballot.sync (predicate depends on previous ballot)
__global__ void kernel_vote_ballot_latency(unsigned int *cycles, int iters) {
  int lane = threadIdx.x;
  int pred = (lane == 0);
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned int b = __ballot_sync(FULL_MASK, pred);
    pred = (b != 0);
  }
  CMB_CLOCK_STOP();
  if (!pred)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Latency of elect.sync (repeated convergent election, result feeds mask)
__global__ void kernel_elect_sync_latency(unsigned int *cycles, int iters) {
  unsigned int mask = FULL_MASK;
  unsigned int lane = 0;
  unsigned int elected = 0;
  CMB_CLOCK_START();
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    unsigned int l, e;
    asm volatile(
        ".reg .pred p;\n\t"
        "elect.sync %0|p, %2;\n\t"
        "selp.b32 %1, 1, 0, p;"
        : "=r"(l), "=r"(e)
        : "r"(mask));
    lane = l;
    elected = e;
    // Keep mask dependent on the previous election so the chain isn't folded.
    mask = FULL_MASK ^ (lane & 0u);
  }
  CMB_CLOCK_STOP();
  if ((lane + elected) == 0xdeadbeef)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Latency of redux.sync.add (input depends on previous reduction)
__global__ void kernel_redux_add_latency(unsigned int *cycles, int iters) {
  int lane = threadIdx.x;
  int v = 1;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    int s = __reduce_add_sync(FULL_MASK, v);
    v = s + lane;
  }
  CMB_CLOCK_STOP();
  if (v == 0xdeadbeef)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Branch prediction warm/cold cost: a branch that is false on the first
// iteration and true for all remaining iterations.
__global__ void kernel_branch_cold(unsigned int *cycles, int iters) {
  int lane = threadIdx.x;
  float a = 1.0f + lane * 0.001f;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    if (i > 0) {
      a = a * 1.0001f + 0.5f;
    }
  }
  CMB_CLOCK_STOP();
  if (a < 0.0f)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Branch always taken (warm), same body as above.
__global__ void kernel_branch_warm(unsigned int *cycles, int iters) {
  int lane = threadIdx.x;
  float a = 1.0f + lane * 0.001f;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    if (i >= 0) {
      a = a * 1.0001f + 0.5f;
    }
  }
  CMB_CLOCK_STOP();
  if (a < 0.0f)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Predicated divergence with a varying active-lane ratio.
// active_lanes = threshold: 8 (25%), 16 (50%), 24 (75%).
__global__ void kernel_predicated_divergence(int active_lanes, int iters,
                                             unsigned int *cycles) {
  int lane = threadIdx.x;
  float a = 1.0f + lane * 0.001f;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    if (lane < active_lanes) {
      a = a * 1.0001f + 0.5f;
    }
  }
  CMB_CLOCK_STOP();
  if (a < 0.0f)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// IPC vs active warps per SM for FFMA (ILP=4 per thread).
__global__ void kernel_ipc_ffma(int warps, int iters, unsigned int *cycles) {
  int tid = threadIdx.x;
  int lane = tid % 32;
  float a0 = 1.0f + lane * 0.001f;
  float a1 = 2.0f + lane * 0.001f;
  float a2 = 3.0f + lane * 0.001f;
  float a3 = 4.0f + lane * 0.001f;
  float b = 1.0001f;
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (tid == 0)
    s_t0 = clockVal();
  __syncthreads();
  if (tid < warps * 32) {
    for (int i = 0; i < iters; ++i) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a0) : "f"(b), "f"(b));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a1) : "f"(b), "f"(b));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a2) : "f"(b), "f"(b));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a3) : "f"(b), "f"(b));
    }
  }
  __syncthreads();
  if (tid == 0)
    s_t1 = clockVal();
  __syncthreads();
  if (tid == 0)
    cycles[0] = s_t1 - s_t0;
  if (a0 + a1 + a2 + a3 == 0xdeadbeef)
    cycles[0] = 0;
}

// IPC vs active warps per SM for IADD3 (measured via add.s32, which maps to
// the IADD3 SASS instruction on modern GPUs).
__global__ void kernel_ipc_iadd3(int warps, int iters, unsigned int *cycles) {
  int tid = threadIdx.x;
  int lane = tid % 32;
  int a0 = 1 + lane;
  int a1 = 2 + lane;
  int a2 = 3 + lane;
  int a3 = 4 + lane;
  int b = 1;
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (tid == 0)
    s_t0 = clockVal();
  __syncthreads();
  if (tid < warps * 32) {
    for (int i = 0; i < iters; ++i) {
      asm volatile("add.s32 %0, %0, %1;" : "+r"(a0) : "r"(b));
      asm volatile("add.s32 %0, %0, %1;" : "+r"(a1) : "r"(b));
      asm volatile("add.s32 %0, %0, %1;" : "+r"(a2) : "r"(b));
      asm volatile("add.s32 %0, %0, %1;" : "+r"(a3) : "r"(b));
    }
  }
  __syncthreads();
  if (tid == 0)
    s_t1 = clockVal();
  __syncthreads();
  if (tid == 0)
    cycles[0] = s_t1 - s_t0;
  if (a0 + a1 + a2 + a3 == 0xdeadbeef)
    cycles[0] = 0;
}

// Divergence cost: each thread executes one branch of a switch over tid % ways.
// A small amount of FFMA work is done in each path so divergence is observable.
template <int WAYS>
__global__ void kernel_divergence(int iters, unsigned int *cycles) {
  int lane = threadIdx.x;
  int branch = lane % WAYS;
  float a = 1.0f + lane * 0.001f;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    if (WAYS == 1) {
      a = a * 1.0001f + 0.5f;
    } else {
      switch (branch) {
        case 0:  a = a * 1.0001f + 0.5f; break;
        case 1:  a = a * 1.0002f + 0.25f; break;
        case 2:  a = a * 1.0003f + 0.125f; break;
        case 3:  a = a * 1.0004f + 0.0625f; break;
        case 4:  a = a * 1.0005f + 0.03125f; break;
        case 5:  a = a * 1.0006f + 0.015625f; break;
        case 6:  a = a * 1.0007f + 0.0078125f; break;
        case 7:  a = a * 1.0008f + 0.00390625f; break;
        case 8:  a = a * 1.0009f + 0.001953125f; break;
        case 9:  a = a * 1.0010f + 0.0009765625f; break;
        case 10: a = a * 1.0011f + 0.00048828125f; break;
        case 11: a = a * 1.0012f + 0.000244140625f; break;
        case 12: a = a * 1.0013f + 0.0001220703125f; break;
        case 13: a = a * 1.0014f + 0.00006103515625f; break;
        case 14: a = a * 1.0015f + 0.000030517578125f; break;
        default: a = a * 1.0016f + 0.0000152587890625f; break;
      }
    }
  }
  CMB_CLOCK_STOP();
  if (a < 0.0f)
    t1 = t0;
  cycles[0] = t1 - t0;
}

static double measure_latency(void (*kernel)(unsigned int *, int),
                              const char *label, int iters) {
  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
  kernel<<<1, 32>>>(d_cycles, iters);
  CUDA_CHECK(cudaGetLastError());
  unsigned int h;
  CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_cycles));
  double cyc = (double)h / iters;
  printf("  %-25s: %.2f cycles\n", label, cyc);
  return cyc;
}

template <int WAYS>
static double measure_divergence(const char *label, int iters) {
  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
  kernel_divergence<WAYS><<<1, 32>>>(iters, d_cycles);
  CUDA_CHECK(cudaGetLastError());
  unsigned int h;
  CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_cycles));
  double cyc = (double)h / iters;
  printf("  %-25s: %.2f cycles/iter\n", label, cyc);
  return cyc;
}

void benchmark_warp_collectives() {
  printf("=== Warp Collectives & Divergence ===\n");
  int iters = 1 << 12;

  double vote_lat = measure_latency(kernel_vote_ballot_latency,
                                    "vote.ballot.sync latency", iters);
  double elect_lat = measure_latency(kernel_elect_sync_latency,
                                     "elect.sync latency", iters);
  double redux_lat = measure_latency(kernel_redux_add_latency,
                                     "redux.sync.add latency", iters);

  g_results.log("warp_collectives", "vote_ballot_latency_cycles", vote_lat,
                "cycles");
  g_results.log("warp_collectives", "elect_sync_latency_cycles", elect_lat,
                "cycles");
  g_results.log("warp_collectives", "redux_add_latency_cycles", redux_lat,
                "cycles");

  printf("  Divergence cost model:\n");
  double div1 = measure_divergence<1>("uniform", 1 << 10);
  double div2 = measure_divergence<2>("2-way", 1 << 10);
  double div4 = measure_divergence<4>("4-way", 1 << 10);
  double div8 = measure_divergence<8>("8-way", 1 << 10);
  double div16 = measure_divergence<16>("16-way", 1 << 10);
  double div32 = measure_divergence<32>("32-way", 1 << 10);

  g_results.log("warp_collectives", "divergence_uniform_cyc_per_iter", div1,
                "cycles");
  g_results.log("warp_collectives", "divergence_2way_cyc_per_iter", div2,
                "cycles");
  g_results.log("warp_collectives", "divergence_4way_cyc_per_iter", div4,
                "cycles");
  g_results.log("warp_collectives", "divergence_8way_cyc_per_iter", div8,
                "cycles");
  g_results.log("warp_collectives", "divergence_16way_cyc_per_iter", div16,
                "cycles");
  g_results.log("warp_collectives", "divergence_32way_cyc_per_iter", div32,
                "cycles");

  // Branch prediction warm/cold
  double branch_cold = measure_latency(kernel_branch_cold,
                                       "branch cold", 1 << 12);
  double branch_warm = measure_latency(kernel_branch_warm,
                                       "branch warm", 1 << 12);
  printf("  Branch cold/warm delta: %.2f cycles\n", branch_cold - branch_warm);
  g_results.log("warp_collectives", "branch_cold_cyc_per_iter", branch_cold,
                "cycles");
  g_results.log("warp_collectives", "branch_warm_cyc_per_iter", branch_warm,
                "cycles");

  // Predicated divergence split-ratio
  auto measure_pred = [&](int active, const char *label) {
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    kernel_predicated_divergence<<<1, 32>>>(active, 1 << 12, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h;
    CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cycles));
    double cyc = (double)h / (1 << 12);
    printf("  Predicated divergence %s: %.2f cycles/iter\n", label, cyc);
    return cyc;
  };
  double pred25 = measure_pred(8, "25% active");
  double pred50 = measure_pred(16, "50% active");
  double pred75 = measure_pred(24, "75% active");
  g_results.log("warp_collectives", "pred_split_25_cyc_per_iter", pred25,
                "cycles");
  g_results.log("warp_collectives", "pred_split_50_cyc_per_iter", pred50,
                "cycles");
  g_results.log("warp_collectives", "pred_split_75_cyc_per_iter", pred75,
                "cycles");

  // IPC vs warps/SM scaling
  {
    DeviceInfo info = queryDevice(0);
    int max_warps = info.max_warps_per_sm;
    int max_warps_per_block = info.max_threads_per_block / 32;
    int sweep = std::min(max_warps, max_warps_per_block);
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    int iters = 1 << 10;

    printf("  IPC vs active warps/SM (ILP=4):\n");
    auto sweep_ipc = [&](auto kernel, const char *prefix) {
      for (int w = 1; w <= sweep; ++w) {
        int threads = w * 32;
        kernel<<<1, threads>>>(w, iters, d_cycles);
        CUDA_CHECK(cudaGetLastError());
        unsigned int h;
        CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        double cycles = (double)h;
        // 4 independent instructions per thread per iteration
        long long insts = (long long)w * 32 * iters * 4;
        double ipc = cycles > 0 ? insts / cycles : 0.0;
        if (w == 1 || w == 4 || w == 8 || w == 16 || w == sweep) {
          printf("    %s W=%2d: IPC=%.2f\n", prefix, w, ipc);
          char key[64];
          snprintf(key, sizeof(key), "%s_ipc_w%d", prefix, w);
          g_results.log("warp_collectives", key, ipc, "inst/cycle");
        }
      }
    };
    sweep_ipc(kernel_ipc_ffma, "ffma");
    sweep_ipc(kernel_ipc_iadd3, "iadd3");
    CUDA_CHECK(cudaFree(d_cycles));
  }
}

} // namespace cmb
