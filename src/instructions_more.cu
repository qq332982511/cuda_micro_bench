#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cmb {

static double measure_latency(void (*kernel)(unsigned int *, int),
                              const char *label, int iters) {
  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
  kernel<<<1, 1>>>(d_cycles, iters);
  CUDA_CHECK(cudaGetLastError());
  unsigned int h;
  CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_cycles));
  double cyc = (double)h / iters;
  double ns = cyc * 1e9 / queryDevice(0).clock_rate_khz;
  printf("  %-20s: %.2f cycles (%.2f ns)\n", label, cyc, ns);
  return cyc;
}

// Logic: 3-input arbitrary logic function (LOP3)
__global__ void kernel_latency_lop3(unsigned int *cycles, int iters) {
  unsigned int a = 0xaaaaaaaaU;
  unsigned int b = 0x55555555U;
  unsigned int c = 0x0f0f0f0fU;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned int r;
    // 0x80 = (a & b & c) truth table bit; keeps a data-dependent chain.
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x80;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
    a = r;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Shift left by constant 1
__global__ void kernel_latency_shl_const(unsigned int *cycles, int iters) {
  unsigned int a = 1U;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned int r;
    asm volatile("shl.b32 %0, %1, 1;" : "=r"(r) : "r"(a));
    a = r;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Variable shift left (amount in register)
__global__ void kernel_latency_shl_var(unsigned int *cycles, int iters) {
  unsigned int a = 1U;
  unsigned int amt = 1U;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned int r;
    asm volatile("shl.b32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(amt));
    a = r;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Byte permute with constant selector
__global__ void kernel_latency_prmt(unsigned int *cycles, int iters) {
  unsigned int a = 0x01020304U;
  unsigned int b = 0x05060708U;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned int r;
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;"
                 : "=r"(r) : "r"(a), "r"(b));
    a = r;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Integer division u32 by runtime value
__global__ void kernel_latency_div_u32(unsigned int *cycles, int iters) {
  unsigned int a = 0x00ffffffU;
  unsigned int b = 7U;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned int r;
    asm volatile("div.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    a = r + 1; // +1 keeps divisor changing slowly but chain depends on result
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// INT64 add
__global__ void kernel_latency_add_u64(unsigned int *cycles, int iters) {
  unsigned long long a = 1ULL;
  unsigned long long b = 3ULL;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned long long r;
    asm volatile("add.s64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
    a = r;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeefULL)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// INT64 multiply low
__global__ void kernel_latency_mul_u64(unsigned int *cycles, int iters) {
  unsigned long long a = 0x00ffffffULL;
  unsigned long long b = 7ULL;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned long long r;
    asm volatile("mul.lo.s64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
    a = r;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeefULL)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Type conversion: f32 <-> u32
__global__ void kernel_latency_cvt_f32_u32(unsigned int *cycles, int iters) {
  float a = 12345.0f;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned int r;
    asm volatile("cvt.rzi.u32.f32 %0, %1;" : "=r"(r) : "f"(a));
    float s;
    asm volatile("cvt.rn.f32.u32 %0, %1;" : "=f"(s) : "r"(r));
    a = s + 0.5f;
  }
  CMB_CLOCK_STOP();
  if ((int)a == 0xdeadbeef)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Type conversion: f32 <-> s32
__global__ void kernel_latency_cvt_f32_s32(unsigned int *cycles, int iters) {
  float a = 12345.0f;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    int r;
    asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(r) : "f"(a));
    float s;
    asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(s) : "r"(r));
    a = s + 0.5f;
  }
  CMB_CLOCK_STOP();
  if ((int)a == 0xdeadbeef)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Type conversion: f32 <-> f64
__global__ void kernel_latency_cvt_f32_f64(unsigned int *cycles, int iters) {
  float a = 1.0001f;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    double r;
    asm volatile("cvt.f64.f32 %0, %1;" : "=d"(r) : "f"(a));
    float s;
    asm volatile("cvt.rn.f32.f64 %0, %1;" : "=f"(s) : "d"(r));
    a = s;
  }
  CMB_CLOCK_STOP();
  if ((int)a == 0xdeadbeef)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Type conversion: u32 -> u64
__global__ void kernel_latency_cvt_u64_u32(unsigned int *cycles, int iters) {
  unsigned int a = 0x00ffffffU;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned long long r;
    asm volatile("cvt.u64.u32 %0, %1;" : "=l"(r) : "r"(a));
    unsigned int s = (unsigned int)(r);
    a = s + 1;
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// SIMD integer add: vadd2 (two 16-bit adds packed in 32-bit)
__global__ void kernel_latency_vadd2(unsigned int *cycles, int iters) {
  unsigned int a = 0x00010001U;
  unsigned int b = 0x00010002U;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    a = __vadd2(a, b);
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// SIMD integer add: vadd4 (four 8-bit adds packed in 32-bit)
__global__ void kernel_latency_vadd4(unsigned int *cycles, int iters) {
  unsigned int a = 0x01010101U;
  unsigned int b = 0x01010101U;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    a = __vadd4(a, b);
  }
  CMB_CLOCK_STOP();
  if (a == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Packed FP16x2 FMA
__global__ void kernel_latency_fma_f16x2(unsigned int *cycles, int iters) {
  unsigned int a = 0x3c003c00U; // 1.0, 1.0
  unsigned int b = 0x3c003c00U;
  unsigned int c = 0;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned int r;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r;
  }
  CMB_CLOCK_STOP();
  if (c == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Packed BF16x2 FMA
__global__ void kernel_latency_fma_bf16x2(unsigned int *cycles, int iters) {
  unsigned int a = 0x3f803f80U; // 1.0, 1.0 bf16
  unsigned int b = 0x3f803f80U;
  unsigned int c = 0;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned int r;
    asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r;
  }
  CMB_CLOCK_STOP();
  if (c == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

// Packed FP32x2 FMA
__global__ void kernel_latency_fma_f32x2(unsigned int *cycles, int iters) {
  unsigned long long a = 0x3f8000003f800000ULL;
  unsigned long long b = 0x3f8000003f800000ULL;
  unsigned long long c = 0;
  CMB_CLOCK_START();
  for (int i = 0; i < iters; ++i) {
    unsigned long long r;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;"
                 : "=l"(r) : "l"(a), "l"(b), "l"(c));
    c = r;
  }
  CMB_CLOCK_STOP();
  if ((unsigned int)c == 0xdeadbeefU)
    t1 = t0;
  cycles[0] = t1 - t0;
}

void benchmark_instructions_more() {
  printf("=== More Instruction Latencies ===\n");
  int iters = 1 << 12; // slower ops need fewer iterations

  double lop3 = measure_latency(kernel_latency_lop3, "LOP3", iters);
  double shl_c = measure_latency(kernel_latency_shl_const, "SHL const", iters);
  double shl_v = measure_latency(kernel_latency_shl_var, "SHL var", iters);
  double prmt = measure_latency(kernel_latency_prmt, "PRMT", iters);
  double div = measure_latency(kernel_latency_div_u32, "DIV u32", 1 << 10);
  double add64 = measure_latency(kernel_latency_add_u64, "ADD u64", iters);
  double mul64 = measure_latency(kernel_latency_mul_u64, "MUL u64", iters);
  double cvt_fu = measure_latency(kernel_latency_cvt_f32_u32, "CVT f32<->u32", iters);
  double cvt_fs = measure_latency(kernel_latency_cvt_f32_s32, "CVT f32<->s32", iters);
  double cvt_ff = measure_latency(kernel_latency_cvt_f32_f64, "CVT f32<->f64", iters);
  double cvt_uu = measure_latency(kernel_latency_cvt_u64_u32, "CVT u64<->u32", iters);
  double vadd2 = measure_latency(kernel_latency_vadd2, "VADD2", iters);
  double vadd4 = measure_latency(kernel_latency_vadd4, "VADD4", iters);
  double f16x2 = measure_latency(kernel_latency_fma_f16x2, "FMA f16x2", iters);
  double bf16x2 = measure_latency(kernel_latency_fma_bf16x2, "FMA bf16x2", iters);
  double f32x2 = measure_latency(kernel_latency_fma_f32x2, "FMA f32x2", iters);

  g_results.log("instructions_more", "lop3_latency_cyc", lop3, "cycles");
  g_results.log("instructions_more", "shl_const_latency_cyc", shl_c, "cycles");
  g_results.log("instructions_more", "shl_var_latency_cyc", shl_v, "cycles");
  g_results.log("instructions_more", "prmt_latency_cyc", prmt, "cycles");
  g_results.log("instructions_more", "div_u32_latency_cyc", div, "cycles");
  g_results.log("instructions_more", "add_u64_latency_cyc", add64, "cycles");
  g_results.log("instructions_more", "mul_u64_latency_cyc", mul64, "cycles");
  g_results.log("instructions_more", "cvt_f32_u32_latency_cyc", cvt_fu, "cycles");
  g_results.log("instructions_more", "cvt_f32_s32_latency_cyc", cvt_fs, "cycles");
  g_results.log("instructions_more", "cvt_f32_f64_latency_cyc", cvt_ff, "cycles");
  g_results.log("instructions_more", "cvt_u64_u32_latency_cyc", cvt_uu, "cycles");
  g_results.log("instructions_more", "vadd2_latency_cyc", vadd2, "cycles");
  g_results.log("instructions_more", "vadd4_latency_cyc", vadd4, "cycles");
  g_results.log("instructions_more", "fma_f16x2_latency_cyc", f16x2, "cycles");
  g_results.log("instructions_more", "fma_bf16x2_latency_cyc", bf16x2, "cycles");
  g_results.log("instructions_more", "fma_f32x2_latency_cyc", f32x2, "cycles");
}

} // namespace cmb
