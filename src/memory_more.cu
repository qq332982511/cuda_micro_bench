#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

namespace cmb {

// Streaming stride probe: each warp reads elements separated by 'stride_bytes'.
__global__ void kernel_stride_latency(const float *data, int n, int stride_elems,
                                      int iters, unsigned int *cycles) {
  int tid = threadIdx.x;
  int idx = (tid * stride_elems) % n;
  float acc = 0.0f;
  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    float v;
    asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(data + idx) : "memory");
    acc += v;
    idx = (idx + stride_elems) % n;
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(acc);
  cycles[tid] = t1 - t0;
}

// Coalescing probe: all threads read with a per-thread stride, one warp.
__global__ void kernel_coalescing(const float *data, int n, int stride_elems,
                                  int iters, unsigned int *cycles) {
  int tid = threadIdx.x;
  int idx = (tid * stride_elems) % n;
  float acc = 0.0f;
  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    float v;
    asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(data + idx) : "memory");
    acc += v;
    idx = (idx + 32 * stride_elems) % n; // each iteration all 32 lanes move forward
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(acc);
  cycles[tid] = t1 - t0;
}

// L1 per-warp bandwidth: single warp reads 128-bit vectors from an L1-resident
// buffer (32 KB) repeatedly.
__global__ void kernel_l1_per_warp_bw(const float4 *data, int n, int iters,
                                      unsigned int *cycles) {
  int tid = threadIdx.x;
  int idx = tid % n;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    float4 value;
    asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
                 : "l"(data + idx) : "memory");
    acc.x += value.x; acc.y += value.y; acc.z += value.z; acc.w += value.w;
    idx = (idx + 32) & (n - 1);
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(acc.x + acc.y + acc.z + acc.w);
  cycles[tid] = t1 - t0;
}

// L2 aggregate bandwidth: many warps/SMs stream 128-bit loads from an L2-sized
// buffer (8 MB).  The buffer is larger than L1 so most hits go to L2.
__global__ void kernel_l2_aggregate_bw(const float4 *data, int n, int iters,
                                       float *out) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;
  int idx = (bid * blockDim.x + tid) % n;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int r = 0; r < iters; ++r) {
    float4 value;
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
                 : "l"(data + idx) : "memory");
    acc.x += value.x; acc.y += value.y; acc.z += value.z; acc.w += value.w;
    idx = (idx + nthreads) % n;
  }
  out[bid * blockDim.x + tid] = acc.x + acc.y + acc.z + acc.w;
}

// Async copy latency: one thread copies a single cache line (16 bytes) from
// global to shared memory and waits for completion.
__global__ void kernel_async_copy_latency(const float *src, int iters,
                                          unsigned int *cycles) {
  __shared__ alignas(16) float smem[8];
  int tid = threadIdx.x;
  const float *gptr = src + tid * 4;
  float *sptr = smem + tid * 4;
  CMB_CLOCK_START();
  for (int r = 0; r < iters; ++r) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(static_cast<unsigned int>(__cvta_generic_to_shared(sptr))), "l"(gptr));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
  }
  CMB_CLOCK_STOP();
  CMB_DEPENDENCY_GUARD(smem[0]);
  cycles[0] = t1 - t0;
}

void benchmark_memory_more() {
  DeviceInfo info = queryDevice(0);
  printf("=== Memory Subsystem Probes ===\n");

  size_t bytes = 8ULL * 1024 * 1024;
  float *d_data;
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMemset(d_data, 0, bytes));
  int n = bytes / sizeof(float);

  unsigned int *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, 32 * sizeof(unsigned int)));
  std::vector<unsigned int> h_cycles(32);

  // 1. Cache-line / stride latency probe.
  printf("  Stride latency probe (cycles per streaming iteration):\n");
  int base_iters = 1 << 14;
  for (int stride_bytes = 4; stride_bytes <= 256; stride_bytes *= 2) {
    int stride_elems = stride_bytes / sizeof(float);
    int iters = base_iters;
    // Reduce iterations for large strides to keep total index within n.
    if (stride_elems * 32 * iters > n)
      iters = n / (stride_elems * 32);
    if (iters < 100)
      continue;

    kernel_stride_latency<<<1, 32>>>(d_data, n, stride_elems, iters, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_cycles.data(), d_cycles, 32 * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    unsigned long long sum = 0;
    for (int i = 0; i < 32; ++i)
      sum += h_cycles[i];
    double cyc = (double)sum / 32.0 / iters;
    printf("    stride %3d B -> %.2f cycles/access\n", stride_bytes, cyc);

    char key[64];
    snprintf(key, sizeof(key), "stride_latency_%d_bytes_cyc", stride_bytes);
    g_results.log("memory_more", key, cyc, "cycles");
  }
  // Streaming stride curves combine coalescing, MLP and cache effects; they
  // cannot identify a cache line size. Keep raw measurements only.

  // 2. Coalescing probe: varying per-thread stride, fixed forward step.
  printf("  Coalescing probe (cycles per warp streaming iteration):\n");
  for (int stride_bytes = 4; stride_bytes <= 128; stride_bytes *= 2) {
    int stride_elems = stride_bytes / sizeof(float);
    int iters = base_iters;
    if (stride_elems * 32 * iters > n)
      iters = n / (stride_elems * 32);
    if (iters < 100)
      continue;

    kernel_coalescing<<<1, 32>>>(d_data, n, stride_elems, iters, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_cycles.data(), d_cycles, 32 * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    unsigned long long sum = 0;
    for (int i = 0; i < 32; ++i)
      sum += h_cycles[i];
    double cyc = (double)sum / 32.0 / iters;
    printf("    stride %3d B -> %.2f cycles per warp-load\n", stride_bytes, cyc);

    char key[64];
    snprintf(key, sizeof(key), "coalesce_stride_%d_bytes_cyc", stride_bytes);
    g_results.log("memory_more", key, cyc, "cycles");
  }

  // 3. L1 per-warp bandwidth (single warp, 128-bit reads).
  {
    const size_t l1_bytes = 32 * 1024;
    float4 *d_l1;
    CUDA_CHECK(cudaMalloc(&d_l1, l1_bytes));
    CUDA_CHECK(cudaMemset(d_l1, 0, l1_bytes));
    int l1_n = l1_bytes / sizeof(float4);
    int iters = 1 << 16;
    kernel_l1_per_warp_bw<<<1, 32>>>(d_l1, l1_n, iters, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_cycles.data(), d_cycles, 32 * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    unsigned long long sum = 0;
    for (int i = 0; i < 32; ++i)
      sum += h_cycles[i];
    double cyc = (double)sum / 32.0 / iters;
    double bytes_per_iter = 32.0 * sizeof(float4);
    double bytes_per_cyc = bytes_per_iter / cyc;
    double gbps = bytes_per_cyc * (info.clock_rate_khz * 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("  L1 per-warp BW: %.2f B/cyc, %.2f GiB/s\n", bytes_per_cyc, gbps);
    g_results.log("memory_more", "l1_per_warp_bw_bytes_per_cyc", bytes_per_cyc,
                  "B/cyc");
    g_results.log("memory_more", "l1_per_warp_bw_gibps", gbps, "nominal GiB/s");
    CUDA_CHECK(cudaFree(d_l1));
  }

  // 4. L2 aggregate bandwidth (multi-SM, L2-sized buffer).
  {
    const size_t l2_bytes = std::min<size_t>(8ULL << 20, info.l2_cache_size / 4);
    float4 *d_l2;
    CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
    kernel_init_stream<<<info.sm_count * 4, 256>>>(reinterpret_cast<unsigned int *>(d_l2), l2_bytes / 4);
    CUDA_CHECK(cudaGetLastError());
    int l2_n = l2_bytes / sizeof(float4);
    int blocks = info.sm_count * 4;
    int threads = 256;
    int iters = 1 << 10;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));

    kernel_l2_aggregate_bw<<<blocks, threads>>>(d_l2, l2_n, iters, d_out);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> samples;
    const double total_bytes = (double)blocks * threads * iters * sizeof(float4);
    for (int sample = 0; sample < 6; ++sample) {
      GpuTimer timer;
      timer.start();
      kernel_l2_aggregate_bw<<<blocks, threads>>>(d_l2, l2_n, iters, d_out);
      timer.stop();
      CUDA_CHECK(cudaGetLastError());
      double ms = timer.elapsedMs();
      if (sample) samples.push_back(total_bytes / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0));
    }
    std::sort(samples.begin(), samples.end());
    double gbps = samples[2];
    printf("  L2 aggregate BW: %.2f GiB/s\n", gbps);
    g_results.log("memory_more", "l2_aggregate_bw_gibps", gbps, "GiB/s");
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_l2));
  }

  // 5. Async copy latency (16 B line, single thread).
  {
    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));
    int iters = 1 << 12;
    kernel_async_copy_latency<<<1, 1>>>(d_data, iters, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h;
    CUDA_CHECK(cudaMemcpy(&h, d_cycles, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    double cyc = (double)h / iters;
    printf("  Async copy 16 B latency: %.2f cycles\n", cyc);
    g_results.log("memory_more", "async_copy_16b_latency_cyc", cyc, "cycles");
  }

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_cycles));
}

} // namespace cmb
