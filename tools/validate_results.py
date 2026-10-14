#!/usr/bin/env python3
"""Check full-run coverage and physical sanity; do not treat another GPU as golden data."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", nargs="?", default="results/results.json")
args = parser.parse_args()
data = json.loads(Path(args.path).read_text())
results = data["results"]


def value(section, key):
    return results[section][key]["value"]


required = {
    "topology": "sm_count",
    "clock": "core_clock_mhz",
    "pipeline_depth": "saturation_ilp_ffma",
    "memory_dram": "read_bw_gibps",
    "memory": "dram_latency_cyc",
    "memory_shared": "bandwidth_gibps",
    "memory_cache": "l1_chase_256_kb_cycles",
    "memory_more": "l2_aggregate_bw_gibps",
    "instructions": "dfma_latency_cyc",
    "instructions_extended": "address_sequence_latency_cycles",
    "instructions_more": "cvt_f32_f64_roundtrip_cyc",
    "coissue": "fma_imad_mixed_total_throughput_ipc",
    "tma": "tma_g2s_load_bw_gibps",
    "atomics": "shared_atomic_inc_updates_gops",
    "tensor_core": "mma_m16n8k8_latency_cyc",
    "tensor_core_more": "fp6_fp4_benchmark_implemented",
    "tensor_core_extended": "int8_m16n8k32_throughput_Tops",
    "occupancy": "max_blocks_per_sm_block_1024",
    "warp_shuffle": "shfl_idx_latency_cycles",
    "warp_collectives": "iadd_ipc_w32",
    "barriers": "mbarrier_cycles",
    "pcie": "um_first_touch_end_to_end_us",
    "cluster": "dsmem_read_latency_cyc",
}
# These sections are expected on the GPU under review, not every CUDA target.
assert data["device"] == "NVIDIA GeForce RTX 5060 Ti", "This audit targets RTX 5060 Ti"
assert data["architecture"] == "12.0"
assert data["sm_count"] == 36
assert data["memory_bus_width_bits"] == 128
assert value("topology", "max_warps_per_sm") == 48
assert value("topology", "l2_cache_size_mb") == 32
assert value("topology", "shared_memory_per_block_default_kb") == 48
assert value("topology", "shared_memory_per_block_optin_kb") == 99
for section, key in required.items():
    assert key in results[section], f"Missing full-run metric: {section}.{key}"

for section, metrics in results.items():
    for key, entry in metrics.items():
        v, unit = entry["value"], entry["unit"]
        if unit == "N/A":
            assert v is None, f"Unavailable metric is numerical: {section}.{key}"
            continue
        assert isinstance(v, (int, float)) and math.isfinite(v), (section, key, v)
        assert v >= 0, (section, key, v)
        if "warp-inst/cycle/SM" in unit:
            assert v <= 4.25, (section, key, "exceeds four scheduler issue slots", v)
        if "warp-PTX-ops/nominal-cycle/SM" in unit:
            # Attribute clocks can differ from boost clocks; allow 25% headroom.
            assert v <= 5, (section, key, "check missing work or wrong count", v)

assert value("tma", "tma_1d_load_latency_cyc") > 0
assert value("tma", "tma_g2s_load_bw_gibps") > 0
assert value("memory_dram", "chase_working_set_bytes") >= 8 * 32 * 1024**2
assert value("memory_dram", "chase_loads_per_sample") == value("memory_dram", "chase_working_set_bytes") / 128
assert value("memory_dram", "chase_min_sector_footprint_bytes") > 32 * 1024**2
assert value("memory_dram", "stream_working_set_bytes") > 2 * 32 * 1024**2
assert value("memory_cache", "l2_chase_working_set_bytes") < 32 * 1024**2
assert 0 < value("memory", "dram_read_bw_gbps") <= 448 * 1.10
# 32 shared banks, 4 bytes/bank/cycle; allow nominal/boost clock differences.
shared_limit_gibps = 32 * 4 * data["sm_count"] * data["clock_khz"] * 1000 / 1024**3
assert 0 < value("memory_shared", "bandwidth_gibps") <= shared_limit_gibps * 1.25
assert 0 < value("memory_more", "l1_per_warp_bw_bytes_per_cyc") <= 128
l1, l2, dram = (value("memory", f"{level}_latency_cyc") for level in ("l1", "l2", "dram"))
assert 0 < l1 < l2 < dram, ("hierarchy order", l1, l2, dram)
assert 0 < value("instructions", "fp32_fma_gflops") / 1000 <= value("peak", "fp32_tflops") * 1.25
assert abs(value("peak", "fp32_cuda_cores_peak_tflops") - value("instructions", "fp32_fma_gflops") / 1000) < 1e-5
for section, key, old in [
    ("pipeline_depth", "inferred_ibuffer_entries_ffma", "I-buffer inference"),
    ("memory_more", "detected_cache_line_bytes", "stride-derived cache line"),
    ("instructions", "atoms_add_latency_cyc", "global/shared atomic alias"),
    ("instructions", "hmma_16816_latency_cyc", "wrong MMA shape alias"),
    ("pcie", "um_page_fault_latency_us", "end-to-end/page fault alias"),
]:
    assert key not in results[section], f"Invalid conclusion returned: {old}"
print(f"Full RTX 5060 Ti result passes coverage, units, finite-value and physical-sanity checks ({sum(map(len, results.values()))} metrics).")
