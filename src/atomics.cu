#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

__global__ void kernel_global_atomic_add(int *counter, int iters) {
  for (int i = 0; i < iters; ++i) {
    atomicAdd(counter, 1);
  }
}

__global__ void kernel_shared_atomic_add(int *out, int iters) {
  __shared__ int counter;
  counter = 0;
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    atomicAdd(&counter, 1);
  }
  __syncthreads();
  out[blockIdx.x] = counter;
}

__global__ void kernel_global_atomic_latency(int *counter, int iters,
                                             unsigned int *cycles) {
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    atomicAdd(counter, 1);
  }
  CMB_CLOCK_STOP();
  cycles[0] = t1 - t0;
}

__global__ void kernel_global_atomic_cas_latency(int *counter, int iters,
                                                 unsigned int *cycles) {
  int old = *counter;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    old = atomicCAS(counter, old, old + 1);
  }
  CMB_CLOCK_STOP();
  if (old == 0xdeadbeef) cycles[0] = 0;
  cycles[0] = t1 - t0;
}

__global__ void kernel_global_atomic_exch_latency(int *counter, int iters,
                                                  unsigned int *cycles) {
  int old = 0;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    old = atomicExch(counter, old + i);
  }
  CMB_CLOCK_STOP();
  if (old == 0xdeadbeef) cycles[0] = 0;
  cycles[0] = t1 - t0;
}

__global__ void kernel_global_atomic_min_latency(int *counter, int iters,
                                                 unsigned int *cycles) {
  int old = 0;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    old = atomicMin(counter, old - 1);
  }
  CMB_CLOCK_STOP();
  if (old == 0xdeadbeef) cycles[0] = 0;
  cycles[0] = t1 - t0;
}

// Block-level reduction latency via shared memory tree (proxy for `red.sync`).
__global__ void kernel_block_reduce_latency(int *data, int iters,
                                            unsigned int *cycles) {
  __shared__ int sdata[256];
  int tid = threadIdx.x;
  int val = data[tid];
  __shared__ unsigned int s_t0, s_t1;
  __syncthreads();
  if (tid == 0)
    s_t0 = clockVal();
  __syncthreads();
  for (int r = 0; r < iters; ++r) {
    sdata[tid] = val;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
      if (tid < s)
        sdata[tid] += sdata[tid + s];
      __syncthreads();
    }
    val = sdata[0] + r;
  }
  __syncthreads();
  if (tid == 0)
    s_t1 = clockVal();
  if (tid == 0)
    cycles[0] = s_t1 - s_t0;
  data[tid] = val;
}

__global__ void kernel_global_atomic_max_latency(int *counter, int iters,
                                                 unsigned int *cycles) {
  int old = 0;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    old = atomicMax(counter, old + 1);
  }
  CMB_CLOCK_STOP();
  if (old == 0xdeadbeef) cycles[0] = 0;
  cycles[0] = t1 - t0;
}

static double measure_atomic_latency(void (*kernel)(int *, int, unsigned int *),
                                     const char *label) {
  DeviceInfo info = queryDevice(0);
  int *d_counter;
  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

  int iters = 1 << 12;
  kernel<<<1, 1>>>(d_counter, iters, d_cycles);
  CUDA_CHECK(cudaGetLastError());
  unsigned int h_cycles;
  CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  double cyc = (double)h_cycles / iters;
  double ns = cyc * 1e9 / info.clock_rate_khz;
  printf("  Global %s latency: %.2f cycles (%.2f ns)\n", label, cyc, ns);

  CUDA_CHECK(cudaFree(d_counter));
  CUDA_CHECK(cudaFree(d_cycles));
  return cyc;
}

void benchmark_atomics() {
  DeviceInfo info = queryDevice(0);
  printf("=== Atomic Benchmarks ===\n");

  // 1. Global atomic throughput (all SMs contend on one counter)
  {
    int *d_counter;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    int iters = 1 << 12;
    int threads = 256;
    int blocks = info.sm_count * 4;

    GpuTimer timer;
    timer.start();
    kernel_global_atomic_add<<<blocks, threads>>>(d_counter, iters);
    timer.stop();
    float ms = timer.elapsedMs();

    long long ops = (long long)blocks * threads * iters;
    double global_atomic_tput = ops / (ms * 1e-3) / 1e9;
    printf("  Global atomicAdd throughput (contended): %.2f GOPS/s\n",
           global_atomic_tput);
    g_results.log("atomics", "global_atomic_add_gops", global_atomic_tput,
                  "GOPS/s");
    CUDA_CHECK(cudaFree(d_counter));
  }

  // 2. Shared atomic throughput (per-block counter)
  {
    int *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, info.sm_count * sizeof(int)));

    int iters = 1 << 12;
    int threads = 256;
    int blocks = info.sm_count * 4;

    GpuTimer timer;
    timer.start();
    kernel_shared_atomic_add<<<blocks, threads>>>(d_out, iters);
    timer.stop();
    float ms = timer.elapsedMs();

    long long ops = (long long)blocks * threads * iters;
    double shared_atomic_tput = ops / (ms * 1e-3) / 1e9;
    printf("  Shared atomicAdd throughput: %.2f GOPS/s\n", shared_atomic_tput);
    g_results.log("atomics", "shared_atomic_add_gops", shared_atomic_tput,
                  "GOPS/s");
    CUDA_CHECK(cudaFree(d_out));
  }

  // 3. Atomic latency
  {
    int *d_counter;
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    kernel_global_atomic_latency<<<1, 1>>>(d_counter, 1 << 12, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h_cycles;
    CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double cyc = (double)h_cycles / (1 << 12);
    double ns = cyc * 1e9 / info.clock_rate_khz;
    printf("  Global atomicAdd latency: %.2f cycles (%.2f ns)\n", cyc, ns);
    g_results.log("atomics", "global_atomic_add_latency_cyc", cyc, "cycles");
    g_results.log("atomics", "global_atomic_add_latency_ns", ns, "ns");
    g_results.log("instructions", "atoms_add_latency_cyc", cyc, "cycles");

    CUDA_CHECK(cudaFree(d_counter));
    CUDA_CHECK(cudaFree(d_cycles));
  }

  // 4. Extended atomic latencies (single-thread pointer-chase style)
  double cas_lat = measure_atomic_latency(kernel_global_atomic_cas_latency, "atomicCAS");
  double exch_lat = measure_atomic_latency(kernel_global_atomic_exch_latency, "atomicExch");
  double min_lat = measure_atomic_latency(kernel_global_atomic_min_latency, "atomicMin");
  double max_lat = measure_atomic_latency(kernel_global_atomic_max_latency, "atomicMax");

  g_results.log("atomics", "global_atomic_cas_latency_cyc", cas_lat, "cycles");
  g_results.log("atomics", "global_atomic_exch_latency_cyc", exch_lat, "cycles");
  g_results.log("atomics", "global_atomic_min_latency_cyc", min_lat, "cycles");
  g_results.log("atomics", "global_atomic_max_latency_cyc", max_lat, "cycles");

  // Block-level reduction latency (shared-memory tree, proxy for red.sync).
  {
    int *d_data;
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_data, 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_data, 0, 256 * sizeof(int)));
    int iters = 1 << 10;
    kernel_block_reduce_latency<<<1, 256>>>(d_data, iters, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h_cycles;
    CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double cyc = (double)h_cycles / iters;
    double ns = cyc * 1e9 / info.clock_rate_khz;
    printf("  Block reduce (tree) latency: %.2f cycles (%.2f ns)\n", cyc, ns);
    g_results.log("atomics", "block_reduce_latency_cyc", cyc, "cycles");
    g_results.log("atomics", "block_reduce_latency_ns", ns, "ns");
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_cycles));
  }
}

} // namespace cmb
