#!/usr/bin/env python3
"""Extract key machine-readable parameters from the micro_arch reference HTML.

Usage:
    python3 tools/extract_reference.py \
        micro_arch_ref/nvidia/sm_120/19_appendix_b_machine_readable.html \
        results/reference.json

The reference HTML path defaults to a relative placeholder below.
"""

import html.parser
import json
import os
import sys


class JsonCodeBlockExtractor(html.parser.HTMLParser):
    """Collect text from <code class="language-json"> blocks."""

    def __init__(self):
        super().__init__()
        self.blocks = []
        self._in_code = False
        self._current = []

    def handle_starttag(self, tag, attrs):
        if tag == "code":
            attr_dict = dict(attrs)
            if attr_dict.get("class", "").startswith("language-json"):
                self._in_code = True
                self._current = []

    def handle_endtag(self, tag):
        if tag == "code" and self._in_code:
            self.blocks.append("".join(self._current))
            self._in_code = False
            self._current = []

    def handle_data(self, data):
        if self._in_code:
            self._current.append(data)


def extract_blocks(path):
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    parser = JsonCodeBlockExtractor()
    parser.feed(html)
    return parser.blocks


def merge_blocks(blocks):
    merged = {}
    for text in blocks:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Warning: skipping malformed JSON block: {e}", file=sys.stderr)
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                    merged[k].update(v)
                else:
                    merged[k] = v
    return merged


def get(d, *path, default=None):
    for key in path:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


def extract_key_params(data):
    """Pull the reference values used by the HTML comparison report."""
    ref = {
        "device": get(data, "metadata", "device", default="RTX PRO 5000 72GB"),
        "architecture": get(data, "metadata", "architecture", default="SM_120"),
        "die": get(data, "metadata", "die", default="GB202"),
        "family": get(data, "metadata", "family", default="Blackwell"),
        "measurement_date": get(data, "metadata", "measurement_date", default="2026-06"),
        "sm_count": get(data, "topology", "sm_count"),
        "fp32_cuda_cores_peak_tflops": get(data, "peak_tflops", "fp32_cuda_cores"),
        "fp16_tensor_tflops": get(data, "peak_tflops", "fp16_tensor"),
        "dram_read_bw_gbps": get(data, "dram", "read_bw_measured_gbps"),
        "dram_write_bw_gbps": get(data, "dram", "write_bw_measured_gbps"),
        "dram_latency_cyc": get(data, "dram", "latency_cyc"),
        "l1_latency_cyc": get(data, "l1_cache", "latency_cyc"),
        "l2_latency_cyc": get(data, "l2_cache", "latency_cyc"),
        "shared_memory_latency_cyc": get(data, "shared_memory", "latency_cyc"),
        "fadd_latency_cyc": get(data, "instructions", "floating_point", "FADD", "latency"),
        "ffma_latency_cyc": get(data, "instructions", "floating_point", "FFMA", "latency"),
        "dfma_latency_cyc": get(data, "instructions", "floating_point", "DFMA", "latency"),
        "mufu_rsq_latency_cyc": get(data, "instructions", "special_function", "MUFU_RSQ", "latency"),
        "atoms_add_latency_cyc": get(data, "instructions", "memory", "ATOMS_ADD", "latency"),
        "hmma_16816_latency_cyc": get(data, "instructions", "tensor_core", "HMMA_16816", "latency"),
        "bar_sync_32w_cyc": get(data, "synchronization", "bar_sync_32w_cyc"),
        "shfl_sync_latency_cyc": get(data, "instructions", "warp_shuffle", "SHFL_SYNC", "latency"),
        # Extra normalization/context fields
        "memory_bus_width_bits": get(data, "dram", "bus_width_bits"),
        "l2_cache_size_mb": get(data, "l2_cache", "capacity_mb"),
        "shared_memory_capacity_per_sm_kb": get(data, "shared_memory", "capacity_per_sm_kb"),
        "boost_mhz": get(data, "clock", "boost_mhz"),
        "memory_data_rate_gbps_per_pin": get(data, "dram", "data_rate_gbps_per_pin"),
    }
    return ref


def main():
    if len(sys.argv) >= 2:
        html_path = sys.argv[1]
    else:
        html_path = "micro_arch_ref/nvidia/sm_120/19_appendix_b_machine_readable.html"

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        out_path = "results/reference.json"

    if not os.path.isfile(html_path):
        print(f"Error: reference HTML not found: {html_path}", file=sys.stderr)
        sys.exit(1)

    blocks = extract_blocks(html_path)
    data = merge_blocks(blocks)
    ref = extract_key_params(data)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2)

    print(f"Wrote {len(ref)} reference parameters to {out_path}")


if __name__ == "__main__":
    main()
