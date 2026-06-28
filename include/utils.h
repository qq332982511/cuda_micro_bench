#pragma once

#include "result_logger.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace cmb {

struct DeviceInfo {
  int id;
  char name[256];
  int cc_major;
  int cc_minor;
  int sm_count;
  int max_threads_per_sm;
  int max_threads_per_block;
  int max_blocks_per_sm;
  int regs_per_sm;
  int regs_per_thread;
  size_t shared_mem_per_sm;
  size_t shared_mem_per_block;
  size_t total_global_mem;
  int warp_size;
  int multi_processor_count;
  int clock_rate_khz;
  int memory_clock_khz;
  int mem_bus_width;
  int l2_cache_size;
  int max_warps_per_sm;
};

inline DeviceInfo queryDevice(int dev = 0) {
  DeviceInfo info{};
  info.id = dev;
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  std::copy(prop.name, prop.name + 256, info.name);
  info.cc_major = prop.major;
  info.cc_minor = prop.minor;
  info.sm_count = prop.multiProcessorCount;
  info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  info.max_threads_per_block = prop.maxThreadsPerBlock;
  info.max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
  info.regs_per_sm = prop.regsPerMultiprocessor;
  info.regs_per_thread = prop.regsPerBlock / prop.maxBlocksPerMultiProcessor;
  info.shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
  info.shared_mem_per_block = prop.sharedMemPerBlock;
  info.total_global_mem = prop.totalGlobalMem;
  info.warp_size = prop.warpSize;
  info.multi_processor_count = prop.multiProcessorCount;
  cudaDeviceGetAttribute(&info.clock_rate_khz, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(&info.memory_clock_khz, cudaDevAttrMemoryClockRate,
                         dev);
  info.mem_bus_width = prop.memoryBusWidth;
  info.l2_cache_size = prop.l2CacheSize;
  info.max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;

  g_results.setDevice(info.name,
                      std::to_string(info.cc_major) + "." +
                          std::to_string(info.cc_minor),
                      info.sm_count, info.clock_rate_khz,
                      info.memory_clock_khz, info.mem_bus_width);
  return info;
}

inline void printDeviceInfo(const DeviceInfo &info) {
  printf("=== Device Info ===\n");
  printf("Device ID           : %d\n", info.id);
  printf("Name                : %s\n", info.name);
  printf("Compute Capability  : %d.%d\n", info.cc_major, info.cc_minor);
  printf("SM Count            : %d\n", info.sm_count);
  printf("Warp Size           : %d\n", info.warp_size);
  printf("Max Warps/SM        : %d\n", info.max_warps_per_sm);
  printf("Max Threads/SM      : %d\n", info.max_threads_per_sm);
  printf("Max Threads/Block   : %d\n", info.max_threads_per_block);
  printf("Max Blocks/SM       : %d\n", info.max_blocks_per_sm);
  printf("Registers/SM        : %d\n", info.regs_per_sm);
  printf("Registers/Thread    : %d\n", info.regs_per_thread);
  printf("Shared Memory/SM    : %zu KB\n", info.shared_mem_per_sm / 1024);
  printf("Shared Memory/Block : %zu KB\n", info.shared_mem_per_block / 1024);
  printf("Global Memory       : %.2f GB\n",
         info.total_global_mem / (1024.0 * 1024.0 * 1024.0));
  printf("Core Clock          : %.3f GHz\n", info.clock_rate_khz / 1e6);
  printf("Memory Clock        : %.3f GHz\n", info.memory_clock_khz / 1e6);
  printf("Memory Bus Width    : %d bit\n", info.mem_bus_width);
  printf("L2 Cache Size       : %d KB\n", info.l2_cache_size / 1024);
  printf("\n");
}

inline int getArchArg() {
  int dev;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  return prop.major * 10 + prop.minor;
}

inline double bandwidthGiB(double bytes, double ns) {
  return bytes / ns * 1e9 / (1024.0 * 1024.0 * 1024.0);
}

} // namespace cmb
