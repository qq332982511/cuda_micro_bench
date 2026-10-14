#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

// ILP sweep kernel: run `ILP` independent dependency chains in one warp.
template <int ILP>
__global__ void kernel_ilp_ffma(int iters, unsigned int *out) {
  float a[ILP];
  for (int i = 0; i < ILP; ++i)
    a[i] = 1.0f + (i + threadIdx.x) * 0.001f;
  const float b = 1.0001f;

  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
#pragma unroll
    for (int i = 0; i < ILP; ++i) {
      float v;
      asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                   : "=f"(v)
                   : "f"(a[i]), "f"(b), "f"(a[i]));
      a[i] = v;
    }
  }
  CMB_CLOCK_STOP();

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ILP; ++i)
    sum += a[i];
  CMB_DEPENDENCY_GUARD(sum);
  if (threadIdx.x == 0) out[0] = t1 - t0;
}

template <int ILP>
__global__ void kernel_ilp_imad(int iters, unsigned int *out) {
  int a[ILP];
  for (int i = 0; i < ILP; ++i)
    a[i] = 1 + i + threadIdx.x;
  // Runtime multiplicand/addend prevent closed-form folding of the loop.
  const int m = threadIdx.x + 3;
  const int n = blockIdx.x + 7;

  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
#pragma unroll
    for (int i = 0; i < ILP; ++i) {
      int v;
      asm volatile("mad.lo.s32 %0, %1, %2, %3;"
                   : "=r"(v)
                   : "r"(a[i]), "r"(m), "r"(n));
      a[i] = v;
    }
  }
  CMB_CLOCK_STOP();

  unsigned int sum = 0;
#pragma unroll
  for (int i = 0; i < ILP; ++i)
    sum += a[i];
  CMB_DEPENDENCY_GUARD(sum);
  if (threadIdx.x == 0) out[0] = t1 - t0;
}

template <typename Kernel>
static double measure_ilp(Kernel kernel, const char *label, int ilp, int iters) {
  unsigned int *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(unsigned int)));
  kernel<<<1, 32>>>(iters, d_out);
  CUDA_CHECK(cudaGetLastError());
  unsigned int h;
  CUDA_CHECK(cudaMemcpy(&h, d_out, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  double total_ops = (double)ilp * iters;
  double cyc_per_op = (double)h / total_ops;
  printf("  %s ILP=%2d: %.3f cyc/op (%.3f op/cyc)\n",
         label, ilp, cyc_per_op, 1.0 / cyc_per_op);
  return cyc_per_op;
}

void benchmark_pipeline_depth() {
  DeviceInfo info = queryDevice(0);
  printf("=== Pipeline Depth / ILP Sweeps ===\n");

  const int iters = 1 << 14;

  // FFMA sweep.
  struct { int ilp; double cyc; } ffma[] = {
      {1, measure_ilp(kernel_ilp_ffma<1>, "FFMA", 1, iters)},
      {2, measure_ilp(kernel_ilp_ffma<2>, "FFMA", 2, iters)},
      {4, measure_ilp(kernel_ilp_ffma<4>, "FFMA", 4, iters)},
      {8, measure_ilp(kernel_ilp_ffma<8>, "FFMA", 8, iters)},
      {16, measure_ilp(kernel_ilp_ffma<16>, "FFMA", 16, iters)},
  };
  for (auto &p : ffma) {
    char key[64];
    snprintf(key, sizeof(key), "ffma_ilp%d_cyc_per_op", p.ilp);
    g_results.log("pipeline_depth", key, p.cyc, "cycles");
  }

  // IMAD sweep.
  struct { int ilp; double cyc; } imad[] = {
      {1, measure_ilp(kernel_ilp_imad<1>, "IMAD", 1, iters)},
      {2, measure_ilp(kernel_ilp_imad<2>, "IMAD", 2, iters)},
      {4, measure_ilp(kernel_ilp_imad<4>, "IMAD", 4, iters)},
      {8, measure_ilp(kernel_ilp_imad<8>, "IMAD", 8, iters)},
      {16, measure_ilp(kernel_ilp_imad<16>, "IMAD", 16, iters)},
  };
  for (auto &p : imad) {
    char key[64];
    snprintf(key, sizeof(key), "imad_ilp%d_cyc_per_op", p.ilp);
    g_results.log("pipeline_depth", key, p.cyc, "cycles");
  }

  // Dependency cost includes loop overhead; it does not measure physical stage count.
  double ffma_depth = ffma[0].cyc;
  double imad_depth = imad[0].cyc;
  printf("  Dependent chain cost: FFMA=%.1f cyc, IMAD=%.1f cyc\n",
         ffma_depth, imad_depth);
  g_results.log("pipeline_depth", "ffma_dependent_chain_cycles", ffma_depth, "cycles");
  g_results.log("pipeline_depth", "imad_dependent_chain_cycles", imad_depth, "cycles");

  // ILP saturation observation (does not measure I-buffer depth): smallest ILP where cycles/op is within 10% of the
  // minimum observed value.
  auto infer_saturation = [](const auto *arr, int n) {
    double min_cyc = arr[0].cyc;
    for (int i = 1; i < n; ++i)
      if (arr[i].cyc < min_cyc)
        min_cyc = arr[i].cyc;
    for (int i = 0; i < n; ++i) {
      if (arr[i].cyc <= min_cyc * 1.10)
        return arr[i].ilp;
    }
    return arr[n - 1].ilp;
  };
  int ffma_saturation = infer_saturation(ffma, 5);
  int imad_saturation = infer_saturation(imad, 5);
  printf("  ILP within 10%% of best measured throughput: FFMA=%d chains, IMAD=%d chains\n",
         ffma_saturation, imad_saturation);
  g_results.log("pipeline_depth", "saturation_ilp_ffma",
                (double)ffma_saturation, "chains");
  g_results.log("pipeline_depth", "saturation_ilp_imad",
                (double)imad_saturation, "chains");
}

} // namespace cmb
