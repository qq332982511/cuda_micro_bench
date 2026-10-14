#include "memory_chase.h"

namespace cmb {
void benchmark_cache() {
  DeviceInfo info = queryDevice(0);
  printf("=== Cache Benchmarks (random dependent loads, median of 5) ===\n");
  printf("  L1 capacity sweep (.ca, one thread, full 128 B slot cycle):\n");
  for (int kb = 4; kb <= 256; kb *= 2) {
    double cyc = measureRandomChase<false>(kb * 1024, true);
    printf("    %4d KiB -> %.2f cycles/access\n", kb, cyc);
    char key[64];
    snprintf(key, sizeof(key), "l1_chase_%d_kb_cycles", kb);
    g_results.log("memory_cache", key, cyc, "cycles/access");
    if (kb == 4) g_results.log("memory", "l1_latency_cyc", cyc, "cycles");
  }
  // Use a working set well above L1 but below L2; .cg excludes L1 hits.
  size_t l2_bytes = std::min<size_t>(8ULL << 20, info.l2_cache_size / 4);
  if (l2_bytes >= (1ULL << 20)) {
    double cyc = measureRandomChase<true>(l2_bytes, true);
    printf("  L2 (.cg, %zu KiB): %.2f cycles/access\n", l2_bytes / 1024, cyc);
    g_results.log("memory", "l2_latency_cyc", cyc, "cycles");
    g_results.log("memory_cache", "l2_chase_working_set_bytes", l2_bytes, "bytes");
  }
}
} // namespace cmb
