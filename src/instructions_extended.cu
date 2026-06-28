#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

// Latency helper: single-thread kernel with dependent chain.
// The sink is stored to global memory at the end to prevent DCE.
// IADD3 has no direct PTX mnemonic; on sm_70+ add.s32 is lowered to IADD3
// (third input tied to RZ), so this measures the same execution unit.
__global__ void kernel_latency_iadd3(int *out, int iters) {
  int a = 1, b = 2;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    int r;
    asm volatile("add.s32 %0, %1, %2;"
                 : "=r"(r)
                 : "r"(a), "r"(b));
    a = r;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeef)
    t1 = clockVal();
  out[0] = t1 - t0;
  out[1] = a;
}

__global__ void kernel_latency_imad(int *out, int iters) {
  int a = 2, b = 7, c = 11;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    int r;
    asm volatile("mad.lo.s32 %0, %1, %2, %3;"
                 : "=r"(r)
                 : "r"(a), "r"(b), "r"(c));
    a = r;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeef)
    t1 = clockVal();
  out[0] = t1 - t0;
  out[1] = a;
}

// LEA-style effective-address: a + b*4 + c.
__global__ void kernel_latency_lea(int *out, int iters) {
  int a = 1, b = 2, c = 3;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    int r;
    asm volatile("mad.lo.s32 %0, %2, 4, %1;\n\t"
                 "add.s32 %0, %0, %3;"
                 : "=r"(r)
                 : "r"(a), "r"(b), "r"(c));
    a = r;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeef)
    t1 = clockVal();
  out[0] = t1 - t0;
  out[1] = a;
}

// IDP4A: 4-element 8-bit integer dot-product accumulate.
__global__ void kernel_latency_idp4a(int *out, int iters) {
  int a = 0x01020304, b = 0x05060708, c = 0;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    int r;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(r)
                 : "r"(a), "r"(b), "r"(c));
    a = r;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeef)
    t1 = clockVal();
  out[0] = t1 - t0;
  out[1] = a;
}

static double measure_inst_latency(void (*kernel)(int *, int), const char *label,
                                   int iters) {
  DeviceInfo info = queryDevice(0);
  int *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(int)));

  kernel<<<1, 1>>>(d_out, iters);
  CUDA_CHECK(cudaGetLastError());

  int h_out[2];
  CUDA_CHECK(
      cudaMemcpy(h_out, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost));

  double cyc = (double)h_out[0] / iters;
  double ns = cyc * 1e9 / info.clock_rate_khz;
  printf("  %-10s latency: %.2f cycles (%.2f ns)\n", label, cyc, ns);
  CUDA_CHECK(cudaFree(d_out));
  return cyc;
}

void benchmark_instructions_extended() {
  DeviceInfo info = queryDevice(0);
  printf("=== Extended Instruction Latency ===\n");

  int cc = info.cc_major * 10 + info.cc_minor;
  if (cc < 70) {
    printf("  IADD3/IMAD/LEA/IDP4A benchmarks target sm_70+. Skipping.\n");
    return;
  }

  int iters = 1 << 14;
  double cyc;

  cyc = measure_inst_latency(kernel_latency_iadd3, "IADD3", iters);
  g_results.log("instructions_extended", "iadd3_latency_cycles", cyc, "cycles");

  cyc = measure_inst_latency(kernel_latency_imad, "IMAD", iters);
  g_results.log("instructions_extended", "imad_latency_cycles", cyc, "cycles");

  cyc = measure_inst_latency(kernel_latency_lea, "LEA", iters);
  g_results.log("instructions_extended", "lea_latency_cycles", cyc, "cycles");

  cyc = measure_inst_latency(kernel_latency_idp4a, "IDP4A", iters);
  g_results.log("instructions_extended", "idp4a_latency_cycles", cyc, "cycles");
}

} // namespace cmb
