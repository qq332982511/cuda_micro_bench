#include "bench_common.h"
#include <cooperative_groups.h>

namespace cmb {
namespace cg = cooperative_groups;
__global__ void kernel_probe_cluster(int *out) {
  auto cluster = cg::this_cluster();
  if (threadIdx.x == 0) out[cluster.block_rank()] = cluster.num_blocks();
  cluster.sync();
}
__global__ void kernel_cluster_barrier_latency(int iters, unsigned int *cycles) {
  auto cluster = cg::this_cluster();
  unsigned int t0 = clockVal();
  for (int i = 0; i < iters; ++i) cluster.sync();
  unsigned int t1 = clockVal();
  // Each block has its own clock domain and its own output slot.
  if (threadIdx.x == 0) cycles[cluster.block_rank()] = t1 - t0;
}
__global__ void kernel_dsmem_latency(unsigned int *cycles) {
  __shared__ unsigned int buf;
  auto cluster = cg::this_cluster();
  if (threadIdx.x == 0) buf = 0;
  cluster.sync();
  if (cluster.block_rank() == 1 && threadIdx.x == 0) {
    unsigned int *remote = cluster.map_shared_rank(&buf, 0);
    unsigned int base = __cvta_generic_to_shared(remote);
    unsigned int idx = 0;
    unsigned int t0 = clockVal();
#pragma unroll 32
    for (int i = 0; i < 4096; ++i) {
      asm volatile("ld.shared::cluster.u32 %0, [%1];"
                   : "=r"(idx) : "r"(base + idx * 4U) : "memory");
    }
    unsigned int t1 = clockAfter(idx);
    cycles[0] = t1 - t0;
  }
  // Remote memory must remain alive until every DSMEM access completes.
  cluster.sync();
}
void benchmark_cluster() {
  queryDevice(0);
  printf("=== Cluster ===\n");
  int supported = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, 0));
  g_results.log("cluster", "launch_supported", supported, "bool");
  if (!supported) { printf("  Cluster launch unsupported.\n"); return; }
  cudaLaunchConfig_t config{};
  config.blockDim = dim3(128);
  config.gridDim = dim3(8);
  int max_size = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(&max_size, kernel_probe_cluster, &config));
  // This is the supported size for this kernel/configuration, not a guessed limit.
  g_results.log("cluster", "max_cluster_size", max_size, "blocks");
  int *out;
  CUDA_CHECK(cudaMalloc(&out, max_size * sizeof(int)));
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {static_cast<unsigned int>(max_size), 1, 1};
  config.attrs = &attr; config.numAttrs = 1; config.gridDim = dim3(max_size);
  CUDA_CHECK(cudaLaunchKernelEx(&config, kernel_probe_cluster, out));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> counts(max_size);
  CUDA_CHECK(cudaMemcpy(counts.data(), out, max_size * sizeof(int), cudaMemcpyDeviceToHost));
  for (int count : counts) if (count != max_size) { fprintf(stderr, "Cluster size mismatch\n"); exit(EXIT_FAILURE); }
  CUDA_CHECK(cudaFree(out));
  printf("  Cluster launch verified: size=%d\n", max_size);
  g_results.log("cluster", "launch_test_size", max_size, "blocks");
  auto measure_barrier = [&](int size, const char *key) {
    unsigned int *cycles;
    CUDA_CHECK(cudaMalloc(&cycles, size * sizeof(unsigned int)));
    config.gridDim = dim3(size); attr.val.clusterDim.x = size;
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel_cluster_barrier_latency, 1024, cycles));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned int> h(size);
    CUDA_CHECK(cudaMemcpy(h.data(), cycles, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    double sum = 0; for (auto v : h) sum += v;
    double cyc = sum / size / 1024;
    printf("  Cluster barrier (%d blocks): %.2f cycles\n", size, cyc);
    g_results.log("cluster", key, cyc, "cycles");
    CUDA_CHECK(cudaFree(cycles));
  };
  if (max_size >= 2) measure_barrier(2, "cluster_barrier_2block_cyc");
  measure_barrier(max_size, "cluster_barrier_max_cyc");
  if (max_size >= 2) {
    unsigned int *cycles;
    CUDA_CHECK(cudaMalloc(&cycles, sizeof(unsigned int)));
    config.gridDim = dim3(2); attr.val.clusterDim.x = 2;
    std::vector<unsigned int> samples;
    for (int sample = 0; sample < 6; ++sample) {
      CUDA_CHECK(cudaLaunchKernelEx(&config, kernel_dsmem_latency, cycles));
      CUDA_CHECK(cudaDeviceSynchronize());
      unsigned int h;
      CUDA_CHECK(cudaMemcpy(&h, cycles, sizeof(h), cudaMemcpyDeviceToHost));
      if (sample) samples.push_back(h);
    }
    std::sort(samples.begin(), samples.end());
    double cyc = samples[2] / 4096.0;
    printf("  DSMEM dependent remote read: %.2f cycles\n", cyc);
    g_results.log("cluster", "dsmem_read_latency_cyc", cyc, "cycles");
    CUDA_CHECK(cudaFree(cycles));
  }
}
} // namespace cmb
