#include "bench_common.h"
#include "utils.h"
#include <cstring>
#include <cstdio>

namespace cmb {

// Benchmark entry points
void benchmark_device_info();
void benchmark_topology_extended();
void benchmark_pipeline_depth();
void benchmark_global_memory();
void benchmark_memory_more();
void benchmark_shared_memory();
void benchmark_cache();
void benchmark_compute_throughput();
void benchmark_compute_latency();
void benchmark_coissue();
void benchmark_tma();
void benchmark_instruction_throughput();
void benchmark_atomics();
void benchmark_tensor_core();
void benchmark_tensor_more();
void benchmark_occupancy();
void benchmark_warp_shuffle();
void benchmark_warp_collectives();
void benchmark_barriers();
void benchmark_instructions_extended();
void benchmark_instructions_more();
void benchmark_tensor_core_extended();
void benchmark_pcie();
void benchmark_cluster();

static void run_all() {
  benchmark_device_info();
  benchmark_topology_extended();
  benchmark_pipeline_depth();
  benchmark_global_memory();
  benchmark_memory_more();
  benchmark_shared_memory();
  benchmark_cache();
  benchmark_compute_throughput();
  benchmark_compute_latency();
  benchmark_coissue();
  benchmark_tma();
  benchmark_instruction_throughput();
  benchmark_atomics();
  benchmark_tensor_core();
  benchmark_tensor_more();
  benchmark_occupancy();
  benchmark_warp_shuffle();
  benchmark_warp_collectives();
  benchmark_barriers();
  benchmark_instructions_extended();
  benchmark_instructions_more();
  benchmark_tensor_core_extended();
  benchmark_pcie();
  benchmark_cluster();
}

static void write_json_results() {
  g_results.writeJson("results/results.json");
}

static void usage(const char *name) {
  printf("Usage: %s [option]...\n", name);
  printf("Options:\n");
  printf("  --device     Device / SM info\n");
  printf("  --topology   Extended topology probes (SMSP, warp scaling)\n");
  printf("  --pipeline   Dependency / ILP sweeps\n");
  printf("  --memory     Global memory bandwidth and latency\n");
  printf("  --memory-more Cache line, coalescing, async-copy probes\n");
  printf("  --shared     Shared memory bandwidth, latency and bank conflicts\n");
  printf("  --cache      Cache size / latency / line-size probe\n");
  printf("  --compute    Compute throughput\n");
  printf("  --latency    Compute latency\n");
  printf("  --coissue    Pipe-to-pipe co-issue overlap\n");
  printf("  --tma        TMA (cp.async.bulk.tensor) latency\n");
  printf("  --thp        Instruction throughput (inst/cycle)\n");
  printf("  --atomics    Atomic throughput and latency\n");
  printf("  --tensor     Tensor Core (MMA) latency and throughput\n");
  printf("  --tensor-more More Tensor Core shapes (FP64 DMMA)\n");
  printf("  --occupancy  Occupancy and launch limits\n");
  printf("  --shuffle    Warp shuffle latency and throughput\n");
  printf("  --collectives Warp collectives (vote/elect/redux) and divergence\n");
  printf("  --barriers   Barrier latencies\n");
  printf("  --instructions Extended instruction latency (PTX ADD, IMAD, address arithmetic, IDP4A)\n");
  printf("  --more-insn  More instruction latencies (logic/shift/prmt/div/conversions)\n");
  printf("  --tensor-ext Extended Tensor Core (FP16/BF16/FP8/INT8 MMA)\n");
  printf("  --pcie       PCIe bandwidth and kernel launch latency\n");
  printf("  --cluster    Thread block cluster probe\n");
  printf("  --json       Run all benchmarks and write results/results.json\n");
  printf("  --all        Run all benchmarks (default)\n");
}

static bool dispatch(const char *opt) {
  if (strcmp(opt, "--device") == 0)
    benchmark_device_info();
  else if (strcmp(opt, "--topology") == 0)
    benchmark_topology_extended();
  else if (strcmp(opt, "--pipeline") == 0)
    benchmark_pipeline_depth();
  else if (strcmp(opt, "--memory") == 0)
    benchmark_global_memory();
  else if (strcmp(opt, "--memory-more") == 0)
    benchmark_memory_more();
  else if (strcmp(opt, "--shared") == 0)
    benchmark_shared_memory();
  else if (strcmp(opt, "--cache") == 0)
    benchmark_cache();
  else if (strcmp(opt, "--compute") == 0)
    benchmark_compute_throughput();
  else if (strcmp(opt, "--latency") == 0)
    benchmark_compute_latency();
  else if (strcmp(opt, "--coissue") == 0)
    benchmark_coissue();
  else if (strcmp(opt, "--tma") == 0)
    benchmark_tma();
  else if (strcmp(opt, "--thp") == 0)
    benchmark_instruction_throughput();
  else if (strcmp(opt, "--atomics") == 0)
    benchmark_atomics();
  else if (strcmp(opt, "--tensor") == 0)
    benchmark_tensor_core();
  else if (strcmp(opt, "--tensor-more") == 0)
    benchmark_tensor_more();
  else if (strcmp(opt, "--occupancy") == 0)
    benchmark_occupancy();
  else if (strcmp(opt, "--shuffle") == 0)
    benchmark_warp_shuffle();
  else if (strcmp(opt, "--collectives") == 0)
    benchmark_warp_collectives();
  else if (strcmp(opt, "--barriers") == 0)
    benchmark_barriers();
  else if (strcmp(opt, "--instructions") == 0)
    benchmark_instructions_extended();
  else if (strcmp(opt, "--more-insn") == 0)
    benchmark_instructions_more();
  else if (strcmp(opt, "--tensor-ext") == 0)
    benchmark_tensor_core_extended();
  else if (strcmp(opt, "--pcie") == 0)
    benchmark_pcie();
  else if (strcmp(opt, "--cluster") == 0)
    benchmark_cluster();
  else if (strcmp(opt, "--all") == 0 || strcmp(opt, "--json") == 0)
    run_all();
  else if (strcmp(opt, "--help") == 0 || strcmp(opt, "-h") == 0)
    usage("cuda_micro_bench");
  else
    return false;
  return true;
}

} // namespace cmb

int main(int argc, char **argv) {
  using namespace cmb;

  if (argc <= 1) {
    run_all();
    write_json_results();
    return 0;
  }

  bool ok = true;
  bool ran = false;
  for (int i = 1; i < argc; ++i) {
    if (!dispatch(argv[i])) {
      printf("Unknown option: %s\n", argv[i]);
      usage(argv[0]);
      ok = false;
    } else {
      ran = true;
    }
  }

  if (ran && ok && g_results.hasDevice()) {
    write_json_results();
  }
  return ok ? 0 : 1;
}
