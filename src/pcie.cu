#include "bench_common.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

namespace cmb {

__global__ void kernel_empty() {}

__global__ void kernel_um_touch(int *p) { *p = *p + 1; }

static double cpu_us(std::chrono::high_resolution_clock::time_point t0,
                     std::chrono::high_resolution_clock::time_point t1) {
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

void benchmark_pcie() {
  DeviceInfo info = queryDevice(0);
  printf("=== PCIe / Launch Latency ===\n");

  // Kernel launch latency (cold): flush, then time first enqueue on CPU.
  {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    kernel_empty<<<1, 1>>>();
    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaGetLastError());
    double us = cpu_us(t0, t1);
    printf("  Kernel launch latency (cold): %.3f us\n", us);
    g_results.log("pcie", "kernel_launch_cold_us", us, "us");
  }

  // Warm the context.
  for (int i = 0; i < 10; ++i) {
    kernel_empty<<<1, 1>>>();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Kernel launch latency (warm): average many launches.
  {
    const int reps = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) {
      kernel_empty<<<1, 1>>>();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double us = cpu_us(t0, t1) / reps;
    printf("  Kernel launch latency (warm, avg of %d): %.3f us\n", reps, us);
    g_results.log("pcie", "kernel_launch_warm_us", us, "us");
  }

  // PCIe bandwidth with pinned buffer.
  const size_t bytes = 64ULL * 1024 * 1024;
  void *h_pinned;
  void *d_buf;
  CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
  CUDA_CHECK(cudaMalloc(&d_buf, bytes));

  // Warm up
  CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Host-to-device bandwidth
  {
    const int reps = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) {
      CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaDeviceSynchronize());
    double us = cpu_us(t0, t1) / reps;
    double gibs = bandwidthGiB(bytes, us * 1e3);
    printf("  HtoD pinned bandwidth: %.2f GiB/s\n", gibs);
    g_results.log("pcie", "htod_bandwidth_GiB_s", gibs, "GiB/s");
  }

  // Device-to-host bandwidth
  {
    const int reps = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) {
      CUDA_CHECK(cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaDeviceSynchronize());
    double us = cpu_us(t0, t1) / reps;
    double gibs = bandwidthGiB(bytes, us * 1e3);
    printf("  DtoH pinned bandwidth: %.2f GiB/s\n", gibs);
    g_results.log("pcie", "dtoh_bandwidth_GiB_s", gibs, "GiB/s");
  }

  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFreeHost(h_pinned));

  // P2P / multi-GPU / Unified Memory availability
  int dev_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&dev_count));
  bool p2p_available = false;
  if (dev_count > 1) {
    int can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, 0, 1));
    p2p_available = (can_access != 0);
  }
  printf("  Device count: %d, P2P available: %s\n", dev_count,
         p2p_available ? "true" : "false");
  g_results.log("pcie", "device_count", (double)dev_count, "");
  g_results.log("pcie", "p2p_available", p2p_available ? 1.0 : 0.0, "bool");

  // Unified Memory page-fault latency: measure a single touch of a managed page.
  // On a single-GPU WSL setup this is essentially a device-memory access.
  double um_fault_us = 0.0;
  if (dev_count >= 1) {
    int *d_managed;
    CUDA_CHECK(cudaMallocManaged(&d_managed, sizeof(int)));
    *d_managed = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    kernel_um_touch<<<1, 1>>>(d_managed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    um_fault_us = cpu_us(t0, t1);
    printf("  Unified Memory first-touch latency: %.3f us\n", um_fault_us);
    CUDA_CHECK(cudaFree(d_managed));
  }
  g_results.log("pcie", "um_page_fault_latency_us", um_fault_us, "us");
}

} // namespace cmb
