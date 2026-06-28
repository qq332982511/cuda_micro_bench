#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

// Use one dynamic shared memory alias per file
extern __shared__ char smem_raw[];

// Shared memory bandwidth: each warp reads/writes shared memory
__global__ void kernel_shared_bw(float *out, int smem_size, int iters) {
  float *smem = reinterpret_cast<float *>(smem_raw);
  int tid = threadIdx.x;
  int idx = tid;
  float acc = 0.0f;

  int n = smem_size / sizeof(float);
  for (int i = tid; i < n; i += blockDim.x) {
    smem[i] = (float)i;
  }
  __syncthreads();

  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    acc += smem[idx];
    idx = (idx + 32) % n;
  }
  CMB_CLOCK_STOP();

  out[blockIdx.x * blockDim.x + tid] = acc;
}

// Shared memory latency using inline PTX
__global__ void kernel_shared_latency(int *out, int iters) {
  int *smem = reinterpret_cast<int *>(smem_raw);
  int tid = threadIdx.x;
  smem[0] = 0;
  __syncthreads();

  int idx = 0;
  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    int v;
    asm volatile("ld.shared.s32 %0, [%1];" : "=r"(v) : "r"(idx));
    idx = v;
  }
  CMB_CLOCK_STOP();

  out[tid] = idx;
  out[blockDim.x + tid] = t1 - t0;
}

// Bank conflict probe: stride access
__global__ void kernel_bank_conflict(const int *in, int *out, int n,
                                     int stride, int iters) {
  int *smem = reinterpret_cast<int *>(smem_raw);
  int tid = threadIdx.x;
  for (int i = tid; i < n; i += blockDim.x)
    smem[i] = in[i];
  __syncthreads();

  int idx = tid * stride;
  int acc = 0;
  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    acc += smem[idx % n];
    idx += 32 * stride;
  }
  CMB_CLOCK_STOP();

  out[tid] = acc;
  out[blockDim.x + tid] = t1 - t0;
}

void benchmark_shared_memory() {
  DeviceInfo info = queryDevice(0);
  printf("=== Shared Memory Benchmarks ===\n");

  printf("  Max Shared Memory/SM: %zu KB\n", info.shared_mem_per_sm / 1024);

  // Shared memory bandwidth
  const int smem_kb = 16;
  const int smem_size = smem_kb * 1024;
  const int iters = 1 << 14;
  int threads = 256;
  int blocks = info.sm_count;

  float *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));

  GpuTimer timer;
  timer.start();
  kernel_shared_bw<<<blocks, threads, smem_size>>>(d_out, smem_size, iters);
  timer.stop();
  CUDA_CHECK(cudaGetLastError());
  float ms = timer.elapsedMs();

  double bytes = (double)blocks * threads * iters * sizeof(float);
  double bw = bandwidthGiB(bytes, ms * 1e6);
  printf("  Shared BW (%d KB, %d iters): %.2f GiB/s\n", smem_kb, iters, bw);
  g_results.log("memory_shared", "bandwidth_gibps", bw, "GiB/s");
  CUDA_CHECK(cudaFree(d_out));

  // Shared memory latency
  int *d_lat;
  CUDA_CHECK(cudaMalloc(&d_lat, 2 * 256 * sizeof(int)));
  kernel_shared_latency<<<1, 256, 1024>>>(d_lat, 1 << 14);
  CUDA_CHECK(cudaGetLastError());
  std::vector<int> h_lat(2 * 256);
  CUDA_CHECK(cudaMemcpy(h_lat.data(), d_lat, 2 * 256 * sizeof(int),
                        cudaMemcpyDeviceToHost));
  long long sum = 0;
  for (int i = 256; i < 2 * 256; ++i)
    sum += h_lat[i];
  double lat_cyc = (double)sum / 256.0 / (1 << 14);
  printf("  Shared latency: %.2f cycles\n", lat_cyc);
  g_results.log("memory_shared", "latency_cycles", lat_cyc, "cycles");
  g_results.log("memory", "shared_memory_latency_cyc", lat_cyc, "cycles");
  CUDA_CHECK(cudaFree(d_lat));

  // Bank conflicts with stride 1..32
  printf("  Bank conflict probe (cycles per access):\n");
  int *d_in, *d_out2;
  CUDA_CHECK(cudaMalloc(&d_in, 4096 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out2, 2 * 256 * sizeof(int)));
  std::vector<int> h_in(4096);
  for (int i = 0; i < 4096; ++i)
    h_in[i] = i;
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), 4096 * sizeof(int),
                        cudaMemcpyHostToDevice));

  for (int stride = 1; stride <= 32; ++stride) {
    kernel_bank_conflict<<<1, 256, 4096 * sizeof(int)>>>(d_in, d_out2, 4096,
                                                         stride, 1 << 12);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_lat.data(), d_out2, 2 * 256 * sizeof(int),
                          cudaMemcpyDeviceToHost));
    long long s = 0;
    for (int i = 256; i < 2 * 256; ++i)
      s += h_lat[i];
    double cyc = (double)s / 256.0 / (1 << 12);
    printf("    stride %2d: %.2f cycles/access\n", stride, cyc);
    if (stride == 1 || stride == 2 || stride == 16 || stride == 17 ||
        stride == 32) {
      char key[64];
      snprintf(key, sizeof(key), "bank_conflict_stride_%d_cycles", stride);
      g_results.log("memory_shared", key, cyc, "cycles/access");
    }
  }
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out2));
}

} // namespace cmb
