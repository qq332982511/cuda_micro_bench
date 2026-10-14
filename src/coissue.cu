#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <vector>

namespace cmb {

static __device__ __forceinline__ unsigned int wallclock() { return clockVal(); }

// Throughput-oriented co-issue kernels.  Saturate the SM and compare the
// mixed IPC to the max single-pipe IPC.

__global__ void kernel_fma_thp(int iters, float *out, unsigned int *cycles) {
  int tid = threadIdx.x;
  float acc[4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) acc[k] = 1.0f + (tid + k) * 0.001f;
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (tid == 0) s_t0 = wallclock();
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int k = 0; k < 4; ++k)
      asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(acc[k]));
  }
  __syncthreads();
  if (tid == 0) s_t1 = wallclock();
  if (tid == 0) cycles[blockIdx.x] = s_t1 - s_t0;
  float s = 0.0f;
  #pragma unroll
  for (int k = 0; k < 4; ++k)
    s += acc[k];
  out[blockIdx.x * blockDim.x + tid] = s;
}

__global__ void kernel_imad_thp(int iters, int *out, unsigned int *cycles) {
  int tid = threadIdx.x;
  int acc[4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) acc[k] = 1 + tid + k;
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (tid == 0) s_t0 = wallclock();
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int k = 0; k < 4; ++k)
      asm volatile("mad.lo.s32 %0, %0, %0, %0;" : "+r"(acc[k]));
  }
  __syncthreads();
  if (tid == 0) s_t1 = wallclock();
  if (tid == 0) cycles[blockIdx.x] = s_t1 - s_t0;
  int s = 0;
  #pragma unroll
  for (int k = 0; k < 4; ++k)
    s += acc[k];
  out[blockIdx.x * blockDim.x + tid] = s;
}

__global__ void kernel_fma_imad_mixed_thp(int iters, float *fout, int *iout,
                                          unsigned int *cycles) {
  int tid = threadIdx.x;
  float facc[4];
  int iacc[4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    facc[k] = 1.0f + (tid + k) * 0.001f;
    iacc[k] = 1 + tid + k;
  }
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (tid == 0) s_t0 = wallclock();
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(facc[k]));
      asm volatile("mad.lo.s32 %0, %0, %0, %0;" : "+r"(iacc[k]));
    }
  }
  __syncthreads();
  if (tid == 0) s_t1 = wallclock();
  if (tid == 0) cycles[blockIdx.x] = s_t1 - s_t0;
  float fs = 0.0f;
  int is = 0;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    fs += facc[k];
    is += iacc[k];
  }
  fout[blockIdx.x * blockDim.x + tid] = fs;
  iout[blockIdx.x * blockDim.x + tid] = is;
}

__global__ void kernel_fma_only(int iters, unsigned int *out) {
  float a = 1.0f + threadIdx.x * 0.001f, b = 1.0001f;
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (threadIdx.x == 0)
    s_t0 = wallclock();
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    float r;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                 : "=f"(r) : "f"(a), "f"(b), "f"(a));
    a = r;
  }
  out[threadIdx.x + 1] = clockAfter(a);
  __syncthreads();
  if (threadIdx.x == 0)
    s_t1 = wallclock();
  __syncthreads();
  if (threadIdx.x == 0)
    out[0] = s_t1 - s_t0;

}

__global__ void kernel_imad_only(int iters, unsigned int *out) {
  int a = 1;
  int m = threadIdx.x + 3;
  int n = blockIdx.x + 7;
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (threadIdx.x == 0)
    s_t0 = wallclock();
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    int r;
    asm volatile("mad.lo.s32 %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a), "r"(m), "r"(n));
    a = r;
  }
  out[threadIdx.x + 1] = clockAfter(a);
  __syncthreads();
  if (threadIdx.x == 0)
    s_t1 = wallclock();
  __syncthreads();
  if (threadIdx.x == 0)
    out[0] = s_t1 - s_t0;

}

__global__ void kernel_fma_imad_mixed(int iters, unsigned int *out) {
  float fa = 1.0f + threadIdx.x * 0.001f, fb = 1.0001f;
  int ia = 1;
  int m = threadIdx.x + 3;
  int n = blockIdx.x + 7;
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (threadIdx.x == 0)
    s_t0 = wallclock();
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    float fr;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                 : "=f"(fr) : "f"(fa), "f"(fb), "f"(fa));
    fa = fr;
    int ir;
    asm volatile("mad.lo.s32 %0, %1, %2, %3;"
                 : "=r"(ir) : "r"(ia), "r"(m), "r"(n));
    ia = ir;
  }
  out[threadIdx.x + 1] = clockAfter(__float_as_uint(fa) ^ static_cast<unsigned int>(ia));
  __syncthreads();
  if (threadIdx.x == 0)
    s_t1 = wallclock();
  __syncthreads();
  if (threadIdx.x == 0)
    out[0] = s_t1 - s_t0;

}

static double measure_cycles(void (*kernel)(int, unsigned int *),
                             const char *label, int iters) {
  unsigned int *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, 33 * sizeof(unsigned int)));
  kernel<<<1, 32>>>(iters, d_out);
  CUDA_CHECK(cudaGetLastError());
  unsigned int h;
  CUDA_CHECK(cudaMemcpy(&h, d_out, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  double cyc = (double)h;
  printf("  %-25s: %.2f cycles (%d iters)\n", label, cyc, iters);
  return cyc;
}

void benchmark_coissue() {
  printf("=== Pipe Co-Issue Probes ===\n");
  int iters = 1 << 14;

  double fma_cyc = measure_cycles(kernel_fma_only, "FMA only", iters);
  double imad_cyc = measure_cycles(kernel_imad_only, "IMAD only", iters);
  double mixed_cyc = measure_cycles(kernel_fma_imad_mixed, "FMA+IMAD mixed", iters);

  double fma_per = fma_cyc / iters;
  double imad_per = imad_cyc / iters;
  double mixed_per_pair = mixed_cyc / iters;
  double max_single = std::max(fma_per, imad_per);
  double min_single = std::min(fma_per, imad_per);
  double overlap = 1.0;
  if (mixed_per_pair > max_single)
    overlap = 1.0 - (mixed_per_pair - max_single) / min_single;
  if (overlap < 0.0)
    overlap = 0.0;

  printf("  FMA cyc/op: %.3f, IMAD cyc/op: %.3f, mixed cyc/pair: %.3f\n",
         fma_per, imad_per, mixed_per_pair);
  printf("  FMA+IMAD overlap ratio: %.2f%%\n", overlap * 100.0);

  g_results.log("coissue", "fma_only_cyc_per_op", fma_per, "cycles");
  g_results.log("coissue", "imad_only_cyc_per_op", imad_per, "cycles");
  g_results.log("coissue", "fma_imad_mixed_cyc_per_pair", mixed_per_pair,
                "cycles");
  g_results.log("coissue", "fma_imad_overlap_ratio", overlap * 100.0, "%");

  // Throughput-based co-issue probe: saturate the SM and compare the mixed
  // IPC to the max single-pipe IPC.  Use one block per SM with 1024 threads
  // so that per-block clock is a good proxy for per-SM wall-clock time.
  {
    DeviceInfo info = queryDevice(0);
    int threads = info.max_threads_per_block; // 1024
    int blocks = info.sm_count;
    int iters = 1 << 12;
    int N = 4;
    float *d_fout;
    int *d_iout;
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_fout, blocks * threads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_iout, blocks * threads * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cycles, blocks * sizeof(unsigned int)));

    long long fma_inst_per_block = (long long)threads * N * iters;
    long long imad_inst_per_block = (long long)threads * N * iters;
    long long mixed_inst_per_block = 2LL * threads * N * iters;

    auto avg_cycles = [&](void (*kernel)(int, float *, unsigned int *)) -> double {
      kernel<<<blocks, threads>>>(iters, d_fout, d_cycles);
      CUDA_CHECK(cudaGetLastError());
      std::vector<unsigned int> h(blocks);
      CUDA_CHECK(cudaMemcpy(h.data(), d_cycles, blocks * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));
      double sum = 0.0;
      for (auto v : h) sum += v;
      return sum / blocks;
    };

    double fma_cyc = avg_cycles(kernel_fma_thp);
    double imad_cyc = 0.0;
    {
      kernel_imad_thp<<<blocks, threads>>>(iters, d_iout, d_cycles);
      CUDA_CHECK(cudaGetLastError());
      std::vector<unsigned int> h(blocks);
      CUDA_CHECK(cudaMemcpy(h.data(), d_cycles, blocks * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));
      double sum = 0.0;
      for (auto v : h) sum += v;
      imad_cyc = sum / blocks;
    }

    double fma_ipc = fma_cyc > 0 ? fma_inst_per_block / 32.0 / fma_cyc : 0.0;
    double imad_ipc = imad_cyc > 0 ? imad_inst_per_block / 32.0 / imad_cyc : 0.0;

    kernel_fma_imad_mixed_thp<<<blocks, threads>>>(iters, d_fout, d_iout,
                                                   d_cycles);
    CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned int> h(blocks);
    CUDA_CHECK(cudaMemcpy(h.data(), d_cycles, blocks * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (auto v : h) sum += v;
    double avg_cyc = sum / blocks;
    double mixed_total_ipc = avg_cyc > 0 ? mixed_inst_per_block / 32.0 / avg_cyc : 0.0;

    double max_single_ipc = std::max(fma_ipc, imad_ipc);
    double ratio = max_single_ipc > 0 ? mixed_total_ipc / max_single_ipc : 0.0;

    printf("  Throughput-based co-issue:\n");
    printf("    FMA IPC: %.2f, IMAD IPC: %.2f\n", fma_ipc, imad_ipc);
    printf("    mixed total IPC: %.2f  (%.1f%% of max single)\n",
           mixed_total_ipc, ratio * 100.0);

    g_results.log("coissue", "fma_only_throughput_ipc", fma_ipc, "warp-inst/cycle/SM");
    g_results.log("coissue", "imad_only_throughput_ipc", imad_ipc, "warp-inst/cycle/SM");
    g_results.log("coissue", "fma_imad_mixed_total_throughput_ipc",
                  mixed_total_ipc, "warp-inst/cycle/SM");
    g_results.log("coissue", "fma_imad_throughput_overlap_ratio",
                  ratio * 100.0, "%");

    CUDA_CHECK(cudaFree(d_fout));
    CUDA_CHECK(cudaFree(d_iout));
    CUDA_CHECK(cudaFree(d_cycles));
  }
}

} // namespace cmb
