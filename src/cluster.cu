#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cmb {

namespace cg = cooperative_groups;

__global__ void kernel_probe_cluster(int *out) {
  auto cluster = cg::this_cluster();
  auto block = cg::this_thread_block();
  if (block.thread_rank() == 0) {
    int cid = cluster.block_rank();
    int csize_x = cluster.dim_threads().x / block.dim_threads().x;
    out[cid] = csize_x;
  }
  cluster.sync();
}

__global__ void kernel_cluster_barrier_latency(int iters,
                                                unsigned int *cycles) {
  auto cluster = cg::this_cluster();
  auto block = cg::this_thread_block();
  unsigned int t0 = 0, t1 = 0;
  if (block.thread_rank() == 0)
    t0 = clockVal();
  for (int i = 0; i < iters; ++i) {
    cluster.sync();
  }
  if (block.thread_rank() == 0)
    t1 = clockVal();
  if (block.thread_rank() == 0)
    cycles[0] = t1 - t0;
}

__global__ void kernel_dsmem_latency(unsigned int *cycles) {
  __shared__ int buf;
  auto cluster = cg::this_cluster();
  auto block = cg::this_thread_block();
  int rank = cluster.block_rank();

  if (rank == 0 && block.thread_rank() == 0) {
    buf = 1;
  }
  cluster.sync();

  if (rank == 1 && block.thread_rank() == 0) {
    int *remote = cluster.map_shared_rank(&buf, 0);
    int sink = 0;
    CMB_CLOCK_START();
    for (int i = 0; i < 1024; ++i) {
      sink += *remote;
    }
    CMB_CLOCK_STOP();
    if (sink < 0)
      t1 = clockVal();
    cycles[0] = t1 - t0;
  }
}

void benchmark_cluster() {
  DeviceInfo info = queryDevice(0);
  printf("=== Cluster ===\n");

  int cc = info.cc_major * 10 + info.cc_minor;
  if (cc < 90) {
    printf("  Thread blocks clusters require compute capability >= 9.0. Skipping.\n");
    return;
  }

  int cluster_supported = 0;
  cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, 0);
  if (cluster_supported == 0) {
    printf("  Cluster launch not supported on this device.\n");
    return;
  }

  // Blackwell/SM_120 reports cluster support; use a conservative fallback
  // because cudaDevAttrMaxClusterSize is not exposed by the runtime API.
  int hw_max = 8;

  printf("  Device reports cluster launch supported, using max size: %d\n", hw_max);
  g_results.log("cluster", "max_cluster_size", hw_max, "blocks");

  // Try launching with max cluster size.
  int *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, hw_max * sizeof(int)));

  cudaLaunchConfig_t config = {};
  config.blockDim.x = 128;
  config.gridDim.x = hw_max;
  config.numAttrs = 1;

  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = hw_max;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;

  cudaError_t err = cudaLaunchKernelEx(&config, kernel_probe_cluster, d_out);
  if (err != cudaSuccess) {
    printf("  Cluster launch (size=%d) failed: %s\n", hw_max,
           cudaGetErrorString(err));
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    int h_out[16];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, hw_max * sizeof(int),
                          cudaMemcpyDeviceToHost));
    printf("  Cluster launch (size=%d) succeeded.\n", hw_max);
    g_results.log("cluster", "launch_test_size", hw_max, "blocks");
  }

  CUDA_CHECK(cudaFree(d_out));

  // Cluster barrier latency for a few cluster sizes.
  auto measure_barrier = [&](int cluster_size, const char *label) {
    unsigned int *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    config.blockDim.x = 128;
    config.gridDim.x = cluster_size;
    attr.val.clusterDim.x = cluster_size;
    err = cudaLaunchKernelEx(&config, kernel_cluster_barrier_latency,
                             1 << 10, d_cycles);
    double cyc = 0.0;
    if (err == cudaSuccess) {
      err = cudaDeviceSynchronize();
      if (err == cudaSuccess) {
        unsigned int h_c;
        CUDA_CHECK(cudaMemcpy(&h_c, d_cycles, sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        cyc = (double)h_c / (1 << 10);
        printf("  Cluster barrier latency (%s, size=%d): %.2f cycles\n",
               label, cluster_size, cyc);
      } else {
        printf("  Cluster barrier (%s) sync failed: %s\n",
               label, cudaGetErrorString(err));
        cudaGetLastError();
      }
    } else {
      printf("  Cluster barrier (%s) launch failed: %s\n",
             label, cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaFree(d_cycles));
    return cyc;
  };

  double barrier2 = measure_barrier(2, "2-block");
  double barrier8 = measure_barrier(hw_max, "max");
  g_results.log("cluster", "cluster_barrier_2block_cyc", barrier2, "cycles");
  g_results.log("cluster", "cluster_barrier_max_cyc", barrier8, "cycles");

  // DSMEM latency: 2 blocks in a cluster, rank-1 reads rank-0 smem.
  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
  config.blockDim.x = 128;
  config.gridDim.x = 2;
  attr.val.clusterDim.x = 2;
  err = cudaLaunchKernelEx(&config, kernel_dsmem_latency, d_cycles);
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
      unsigned int h_c;
      CUDA_CHECK(cudaMemcpy(&h_c, d_cycles, sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));
      double cyc = (double)h_c / 1024.0;
      printf("  DSMEM read latency: %.2f cycles\n", cyc);
      g_results.log("cluster", "dsmem_read_latency_cyc", cyc, "cycles");
    } else {
      printf("  DSMEM latency test skipped (sync failure): %s\n",
             cudaGetErrorString(err));
      cudaGetLastError(); // clear error
    }
  } else {
    printf("  DSMEM latency test skipped (launch failure): %s\n",
           cudaGetErrorString(err));
  }
  // Some devices report cluster launch support but do not support DSMEM
  // remote shared-memory access; in that case the context may be in a sticky
  // error state. Do not abort on free failure here.
  err = cudaFree(d_cycles);
  if (err != cudaSuccess) {
    printf("  (cudaFree returned %s; clearing sticky error)\n",
           cudaGetErrorString(err));
    cudaGetLastError();
  }
}

} // namespace cmb
