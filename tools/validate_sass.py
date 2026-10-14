#!/usr/bin/env python3
"""Verify that measured work remains in sm_120 machine-code loops after ptxas."""
import re
import subprocess

sass = subprocess.check_output(["cuobjdump", "--dump-sass", "cuda_micro_bench"], text=True)
functions = {}
for block in sass.split("Function : ")[1:]:
    symbol = block.splitlines()[0].strip()
    instructions = []
    for line in block.splitlines():
        match = re.search(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*/\* 0x", line)
        if match:
            instructions.append((int(match[1], 16), match[2]))
    functions[symbol] = instructions


def kernel(name):
    # Exact identifier encoding avoids e.g. matching the mixed_thp kernel too.
    matches = [ops for symbol, ops in functions.items() if f"{len(name)}{name}E" in symbol]
    assert len(matches) == 1, (name, "entry registration or symbol lookup", len(matches))
    return matches[0]


def opcode(instruction):
    words = instruction.split()
    return words[1] if words[0].startswith("@") else words[0]


def live_loop(name, instruction_prefix):
    ops = kernel(name)
    for end, instruction in ops:
        branch = re.search(r"\bBRA(?:\.U)?\s+.*?0x([0-9a-f]+)", instruction)
        if branch:
            start = int(branch[1], 16)
            if start < end and any(opcode(op).startswith(instruction_prefix) for addr, op in ops if start <= addr <= end):
                return
    raise AssertionError((name, f"no {instruction_prefix} in a repeated machine-code loop"))


for name, prefix in [
    ("kernel_latency_int32_add", "IADD"),
    ("kernel_latency_iadd3", "IADD"),
    ("kernel_latency_int32_mul", "IMAD"),
    ("kernel_latency_rcp", "MUFU.RCP"),
    ("kernel_latency_ex2", "MUFU.EX2"),
    ("kernel_latency_lg2", "MUFU.LG2"),
    ("kernel_ipc_iadd3", "IADD"),
    ("kernel_ipc_ffma", "FFMA"),
    ("kernel_shared_bw", "LDS.128"),
    ("kernel_sts_thp", "STS"),
    ("kernel_l1_per_warp_bw", "LDG.E.128.STRONG.SM"),
    ("kernel_l2_aggregate_bw", "LDG.E.128.STRONG.GPU"),
    ("kernel_dsmem_latency", "LD.E"),
    ("kernel_latency_fma_f16x2", "HFMA2"),
    ("kernel_latency_fma_bf16x2", "HFMA2"),
    ("kernel_latency_fma_f32x2", "FFMA"),
]:
    live_loop(name, prefix)

for name in ["kernel_fma_thp", "kernel_imad_thp", "kernel_fma_imad_mixed_thp", "kernel_fma_imad_mixed", "kernel_fma_mma_mixed"]:
    assert not any(opcode(op).startswith(("UFFMA", "UIMAD")) for _, op in kernel(name)), (name, "uniform execution path")
for name in ["kernel_latency_int32_add", "kernel_latency_iadd3"]:
    assert not any(opcode(op) == "IADD3" for _, op in kernel(name)), (name, "ADDs fused into IADD3")
    assert any("CS2R" in op and op.startswith("@") for _, op in kernel(name)), (name, "missing predicated end clock")
assert not any("WARPSYNC" in op for _, op in kernel("kernel_syncwarp")), "Update the sync-elision conclusion for this compiler/target"
print("SASS check passes: timed arithmetic/memory loops remain, ADDs are not fused, vector co-issue paths and end-clock predicates are present.")
