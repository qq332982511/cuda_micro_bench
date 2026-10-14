#include "bench_common.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

void benchmark_device_info() {
  DeviceInfo info = queryDevice(0);
  printDeviceInfo(info);

  // Additional derived metrics
  // Rough estimate assuming 128 FP32 CUDA cores per SM (Ampere/Ada/Blackwell).
  double peak_fp32_tflops =
      (double)info.sm_count * 128 * 2 * info.clock_rate_khz / 1e9;
  // Consumer GPUs typically have FP64 throughput ~1/64 of FP32.
  double peak_fp64_tflops = peak_fp32_tflops / 64.0;
  double theoretical_bw =
      (double)info.memory_clock_khz * 2 * info.mem_bus_width / 8.0 / 1e6;

  printf("=== Derived / Theoretical Limits ===\n");
  printf("Peak FP32 (theoretical, all SMs) : %.2f TFLOP/s\n", peak_fp32_tflops);
  printf("Peak FP64 (consumer estimate)    : %.2f TFLOP/s\n", peak_fp64_tflops);
  printf("Peak Memory BW (theoretical)     : %.2f GB/s\n", theoretical_bw);
  printf("\n");

  g_results.log("topology", "sm_count", info.sm_count, "");
  g_results.log("topology", "max_warps_per_sm", info.max_warps_per_sm, "");
  g_results.log("topology", "max_threads_per_sm", info.max_threads_per_sm, "");
  g_results.log("topology", "max_threads_per_block", info.max_threads_per_block, "");
  g_results.log("topology", "max_blocks_per_sm", info.max_blocks_per_sm, "");
  g_results.log("topology", "registers_per_sm", info.regs_per_sm, "");
  g_results.log("topology", "shared_memory_per_sm_kb",
                info.shared_mem_per_sm / 1024.0, "KiB");
  g_results.log("topology", "shared_memory_per_block_default_kb", info.shared_mem_per_block / 1024.0, "KiB");
  g_results.log("topology", "shared_memory_per_block_optin_kb", info.shared_mem_per_block_optin / 1024.0, "KiB");
  g_results.log("topology", "shared_memory_banks", 32, "");
  g_results.log("topology", "shared_memory_bank_width_bytes", 4, "bytes");
  g_results.log("topology", "l2_cache_size_mb", info.l2_cache_size / (1024.0 * 1024.0),
                "MiB");
  g_results.log("clock", "core_clock_mhz", info.clock_rate_khz / 1000.0, "MHz");
  g_results.log("clock", "memory_clock_mhz", info.memory_clock_khz / 1000.0, "MHz");
  g_results.log("peak", "fp32_tflops", peak_fp32_tflops, "TFLOP/s");
  g_results.log("peak", "fp64_tflops", peak_fp64_tflops, "TFLOP/s");
  g_results.log("peak", "memory_bw_theoretical_gbps", theoretical_bw, "GB/s");

  // GPC/TPC/L2 partition/memory controller counts are not exposed by the CUDA
  // Runtime on this platform (no NVML).  Log N/A so the report still has the
  // keys and the checklist can be marked as environment-limited.
  printf("=== Topology Values Not Exposed by Runtime ===\n");
  printf("GPC count             : N/A (not exposed)\n");
  printf("TPC count             : N/A (not exposed)\n");
  printf("L2 partition count    : N/A (not exposed)\n");
  printf("Memory controller count: N/A (not exposed)\n");
  g_results.log("topology", "gpc_count", 0.0, "N/A");
  g_results.log("topology", "tpc_count", 0.0, "N/A");
  g_results.log("topology", "l2_partition_count", 0.0, "N/A");
  g_results.log("topology", "memory_controller_count", 0.0, "N/A");
}

} // namespace cmb
