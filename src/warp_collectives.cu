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
  CMB_DEPENDENCY_GUARD(pred);
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
}

// Repeated convergent elect.sync issue cost; no dependent chain is possible here.
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
    mask = FULL_MASK;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(lane + elected);
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
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
  CMB_DEPENDENCY_GUARD(v);
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
}

// Branch prediction warm/cold cost: a branch that is false on the first
// iteration and true for all remaining iterations.
__global__ void kernel_branch_cold(unsigned int *cycles, int iters) {
  int lane = threadIdx.x;
  float a = 1.0f + lane * 0.001f;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    if (i > 0) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.999f), "f"(0.01f));
    }
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
}

// Branch always taken (warm), same body as above.
__global__ void kernel_branch_warm(unsigned int *cycles, int iters) {
  int lane = threadIdx.x;
  float a = 1.0f + lane * 0.001f;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    if (i >= 0) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.999f), "f"(0.01f));
    }
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
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
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.999f), "f"(0.01f));
    }
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
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
  cycles[tid + 1] = clockAfter(a0 + a1 + a2 + a3);
  __syncthreads();
  if (tid == 0)
    s_t1 = clockVal();
  __syncthreads();
  if (tid == 0)
    cycles[0] = s_t1 - s_t0;

}

// IPC vs active warps per SM for IADD3 (measured via add.s32, which maps to
// the IADD3 SASS instruction on modern GPUs).
__global__ void kernel_ipc_iadd3(int warps, int iters, unsigned int *cycles) {
  int tid = threadIdx.x;
  int a0 = 1 + tid;
  int a1 = 2 + tid;
  int a2 = 3 + tid;
  int a3 = 4 + tid;
  int b = 1 + tid;
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (tid == 0)
    s_t0 = clockVal();
  __syncthreads();
  if (tid < warps * 32) {
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
      asm volatile("add.s32 %0, %0, %1;" : "+r"(a0) : "r"(b));
      asm volatile("add.s32 %0, %0, %1;" : "+r"(a1) : "r"(b));
      asm volatile("add.s32 %0, %0, %1;" : "+r"(a2) : "r"(b));
      asm volatile("add.s32 %0, %0, %1;" : "+r"(a3) : "r"(b));
    }
  }
  cycles[tid + 1] = clockAfter(a0 + a1 + a2 + a3);
  __syncthreads();
  if (tid == 0)
    s_t1 = clockVal();
  __syncthreads();
  if (tid == 0)
    cycles[0] = s_t1 - s_t0;

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
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.999f), "f"(0.01f));
    } else {
      switch (branch) {
        case 0: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9990f), "f"(0.0100f)); break;
        case 1: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9989f), "f"(0.0110f)); break;
        case 2: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9988f), "f"(0.0120f)); break;
        case 3: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9987f), "f"(0.0130f)); break;
        case 4: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9986f), "f"(0.0140f)); break;
        case 5: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9985f), "f"(0.0150f)); break;
        case 6: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9984f), "f"(0.0160f)); break;
        case 7: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9983f), "f"(0.0170f)); break;
        case 8: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9982f), "f"(0.0180f)); break;
        case 9: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9981f), "f"(0.0190f)); break;
        case 10: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9980f), "f"(0.0200f)); break;
        case 11: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9979f), "f"(0.0210f)); break;
        case 12: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9978f), "f"(0.0220f)); break;
        case 13: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9977f), "f"(0.0230f)); break;
        case 14: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9976f), "f"(0.0240f)); break;
        case 15: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9975f), "f"(0.0250f)); break;
        case 16: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9974f), "f"(0.0260f)); break;
        case 17: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9973f), "f"(0.0270f)); break;
        case 18: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9972f), "f"(0.0280f)); break;
        case 19: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9971f), "f"(0.0290f)); break;
        case 20: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9970f), "f"(0.0300f)); break;
        case 21: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9969f), "f"(0.0310f)); break;
        case 22: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9968f), "f"(0.0320f)); break;
        case 23: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9967f), "f"(0.0330f)); break;
        case 24: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9966f), "f"(0.0340f)); break;
        case 25: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9965f), "f"(0.0350f)); break;
        case 26: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9964f), "f"(0.0360f)); break;
        case 27: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9963f), "f"(0.0370f)); break;
        case 28: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9962f), "f"(0.0380f)); break;
        case 29: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9961f), "f"(0.0390f)); break;
        case 30: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9960f), "f"(0.0400f)); break;
        case 31: asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(0.9959f), "f"(0.0410f)); break;
      }
    }
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(a);
  if (threadIdx.x == 0) cycles[0] = t1 - t0;
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
                                     "elect.sync repeated issue", iters);
  double redux_lat = measure_latency(kernel_redux_add_latency,
                                     "redux.sync.add latency", iters);

  g_results.log("warp_collectives", "vote_ballot_latency_cycles", vote_lat,
                "cycles");
  g_results.log("warp_collectives", "elect_sync_issue_cycles", elect_lat,
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
                                       "first iteration skip", 1 << 12);
  double branch_warm = measure_latency(kernel_branch_warm,
                                       "always work", 1 << 12);
  printf("  Different-loop timing delta: %.2f cycles\n", branch_cold - branch_warm);
  g_results.log("warp_collectives", "first_iteration_skip_cyc_per_iter", branch_cold,
                "cycles");
  g_results.log("warp_collectives", "always_work_cyc_per_iter", branch_warm,
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
    CUDA_CHECK(cudaMalloc(&d_cycles, (info.max_threads_per_block + 1) * sizeof(unsigned int)));
    int iters = 1 << 12;

    printf("  IPC vs active warps/SM (ILP=4):\n");
    auto sweep_ipc = [&](auto kernel, const char *prefix) {
      for (int w = 1; w <= sweep; ++w) {
        int threads = w * 32;
        std::vector<unsigned int> samples;
        for (int sample = 0; sample < 6; ++sample) {
          kernel<<<1, threads>>>(w, iters, d_cycles);
          CUDA_CHECK(cudaGetLastError());
          unsigned int h;
          CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int), cudaMemcpyDeviceToHost));
          if (sample) samples.push_back(h);
        }
        std::sort(samples.begin(), samples.end());
        double cycles = samples[2];
        // 4 independent instructions per thread per iteration
        long long insts = (long long)w * 32 * iters * 4;
        double ipc = cycles > 0 ? insts / 32.0 / cycles : 0.0;
        if (w == 1 || w == 4 || w == 8 || w == 16 || w == sweep) {
          printf("    %s W=%2d: IPC=%.2f\n", prefix, w, ipc);
          char key[64];
          snprintf(key, sizeof(key), "%s_ipc_w%d", prefix, w);
          g_results.log("warp_collectives", key, ipc, "warp-inst/cycle/SM");
        }
      }
    };
    sweep_ipc(kernel_ipc_ffma, "ffma");
    sweep_ipc(kernel_ipc_iadd3, "iadd");
    CUDA_CHECK(cudaFree(d_cycles));
  }
}

} // namespace cmb
