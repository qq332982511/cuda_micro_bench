#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <utility>

namespace cmb {

static __device__ __forceinline__ unsigned int wallclock() { return clockVal(); }

// 1D TMA latency kernel: a single warp, one elected thread issues repeated
// cp.async.bulk.tensor.1d.shared::cta.global loads and waits on an mbarrier.
// The tensor map is passed by value as a __grid_constant__ parameter, which is
// the simplest/safest pattern on Hopper/Blackwell.
__global__ void kernel_tma_1d_latency(const __grid_constant__ CUtensorMap tmap,
                                      int iters,
                                      unsigned int *out_cycles) {
  __shared__ alignas(128) char smem_buf[128];
  __shared__ alignas(8) uint64_t mbar;
  __shared__ unsigned int s_t0, s_t1;

  uint32_t smem_addr = __cvta_generic_to_shared(smem_buf);
  uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);

  if (threadIdx.x == 0) {
    // expected arrival count = 1 (only this thread will arrive/issue TMA)
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(mbar_addr), "r"(1));
    // Make the initialized mbarrier visible to the async/TMA proxy.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    s_t0 = wallclock();
    for (int i = 0; i < iters; ++i) {
      // issue 1D TMA load of 16 bytes
      asm volatile("cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                   ::
                   "r"(smem_addr),
                   "l"(&tmap),
                   "r"(0u),
                   "r"(mbar_addr)
                   : "memory");
      // arrive and tell the mbarrier how many bytes the TMA will transfer
      uint64_t token __attribute__((unused));
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared.b64 %0, [%1], %2;"
                   : "=l"(token)
                   : "r"(mbar_addr), "r"(16));
      // Phase parity alternates 0,1,0,1,... starting from 0.
      unsigned int parity = (unsigned int)(i & 1);
      unsigned int ready = 0;
      do {
        asm volatile(
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared.b64 p, [%1], %2;\n\t"
            "selp.b32 %0, 1, 0, p;"
            : "=r"(ready)
            : "r"(mbar_addr), "r"(parity));
      } while (!ready);
    }
    s_t1 = wallclock();
    *out_cycles = s_t1 - s_t0;
  }
}

// TMA global->shared bulk-copy bandwidth configuration.
constexpr int kTmaStages = 4;               // pipeline depth (barriers)
constexpr size_t kTmaChunkBytes = 16 << 10; // 16 KiB per TMA transfer

using tma_barrier = cuda::barrier<cuda::thread_scope_block>;

extern __shared__ char smem_raw[];

// G2S TMA bandwidth kernel: each CTA streams a contiguous slice of global
// memory through kTmaStages shared-memory buffers using the CUDA C++ TMA
// interface (cuda::device::memcpy_async_tx + transaction barriers). A single
// thread drives the pipeline; throughput is timed host-side.
__global__ void kernel_tma_g2s_bandwidth(const char *gmem, size_t slice_bytes,
                                         int chunks_per_block,
                                         unsigned int *sink) {
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ tma_barrier bar[kTmaStages];

  if (threadIdx.x == 0) {
    for (int s = 0; s < kTmaStages; ++s)
      init(&bar[s], 1); // only this thread arrives on each barrier
    // Make the initialized barriers visible to the async/TMA proxy.
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const char *src = gmem + (size_t)blockIdx.x * slice_bytes;
    tma_barrier::arrival_token token[kTmaStages];
    for (int c = 0; c < chunks_per_block; ++c) {
      int stage = c % kTmaStages;
      if (c >= kTmaStages) {
        // Reusing a stage: wait for its previous transfer to complete.
        bar[stage].wait(std::move(token[stage]));
      }
      char *dst = smem_raw + (size_t)stage * kTmaChunkBytes;
      token[stage] = cuda::device::barrier_arrive_tx(
          bar[stage], 1, (cuda::std::ptrdiff_t)kTmaChunkBytes);
      cuda::device::memcpy_async_tx(dst, src + (size_t)c * kTmaChunkBytes,
                                    cuda::aligned_size_t<16>(kTmaChunkBytes),
                                    bar[stage]);
    }
    // Drain the in-flight stages before the block exits.
    int drain = chunks_per_block < kTmaStages ? chunks_per_block : kTmaStages;
    for (int s = 0; s < drain; ++s)
      bar[s].wait(std::move(token[s]));
  }
  __syncthreads();
  // keep smem references alive
  if (threadIdx.x == 0) sink[blockIdx.x] = static_cast<unsigned char>(smem_raw[0]);
}

static bool encode_1d_tensor_map(void *d_buf,
                                 size_t nbytes,
                                 CUtensorMap *h_tmap) {
  std::memset(h_tmap, 0, sizeof(CUtensorMap));

  // cuTensorMapEncodeTiled expects non-null arrays even when rank-1 == 0.
  uint64_t global_dim = nbytes;
  uint64_t global_stride = 0;
  uint32_t box_dim = 16;          // 16-byte tile
  uint32_t elem_stride = 1;

  CUresult res = cuTensorMapEncodeTiled(
      h_tmap,
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      1,                 // rank
      d_buf,
      &global_dim,       // globalDim (rank 1)
      &global_stride,    // globalStrides (rank-1 = 1 element, ignored for rank 1)
      &box_dim,          // boxDim
      &elem_stride,      // elementStrides
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (res != CUDA_SUCCESS) {
    const char *errstr = nullptr;
    cuGetErrorString(res, &errstr);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", errstr ? errstr : "unknown");
    exit(EXIT_FAILURE);
  }
  return true;
}

void benchmark_tma() {
  queryDevice(0);
  printf("=== TMA (cp.async.bulk.tensor) Probes ===\n");

  int cc_major = 0, cc_minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0));
  if (cc_major < 9) {
    printf("  TMA requires SM >= 9.0 (current %d.%d), skipping.\n", cc_major, cc_minor);
    g_results.log("tma", "tma_1d_load_latency_cyc", 0.0, "N/A");
    g_results.log("tma", "tma_g2s_load_bw_gibps", 0.0, "N/A");
    return;
  }

  CUresult drv_res = cuInit(0);
  if (drv_res != CUDA_SUCCESS) {
    const char *errstr = nullptr;
    cuGetErrorString(drv_res, &errstr);
    fprintf(stderr, "cuInit failed: %s\n", errstr ? errstr : "unknown");
    exit(EXIT_FAILURE);
  }

  const size_t gmem_bytes = 1024;
  char *d_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, gmem_bytes));
  CUDA_CHECK(cudaMemset(d_buf, 0, gmem_bytes));

  alignas(64) CUtensorMap h_tmap;
  bool ok = encode_1d_tensor_map(d_buf, gmem_bytes, &h_tmap);

  unsigned int *d_cycles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned int)));

  double latency_cyc = 0.0;
  if (ok) {
    int iters = 256;
    kernel_tma_1d_latency<<<1, 32>>>(h_tmap, iters, d_cycles);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h_cycles;
    CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    latency_cyc = (double)h_cycles / (double)iters;
    printf("  TMA 1D 16-byte load issue/wait cost: %.2f cycles (%u cycles / %d iters)\n",
           latency_cyc, h_cycles, iters);
  }

  if (!ok) latency_cyc = 0.0;
  g_results.log("tma", "tma_1d_load_latency_cyc", latency_cyc, "cycles");

  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_buf));

  // --- TMA global->shared bulk-copy bandwidth ---
  // Each CTA streams a contiguous slice of a large buffer into shared memory
  // with pipelined cp.async.bulk ops; throughput is timed host-side.
  double bw_gib = 0.0;
  {
    DeviceInfo info = queryDevice(0);
    size_t bytes = 256ull << 20;
    if (bytes > info.total_global_mem / 4)
      bytes = info.total_global_mem / 4;
    int blocks = info.sm_count * 8;
    size_t slice = (bytes / (size_t)blocks) / kTmaChunkBytes * kTmaChunkBytes;
    int smem_bytes = (int)(kTmaStages * kTmaChunkBytes);

    if (slice >= (size_t)kTmaStages * kTmaChunkBytes) {
      size_t total = slice * (size_t)blocks;
      char *d_src = nullptr;
      unsigned int *d_sink = nullptr;
      CUDA_CHECK(cudaMalloc(&d_src, total));
      kernel_init_stream<<<blocks, 256>>>(reinterpret_cast<unsigned int *>(d_src), total / 4);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMalloc(&d_sink, blocks * sizeof(unsigned int)));
      CUDA_CHECK(cudaFuncSetAttribute(kernel_tma_g2s_bandwidth,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_bytes));

      int chunks = (int)(slice / kTmaChunkBytes);
      // Warm-up launch and sync.
      kernel_tma_g2s_bandwidth<<<blocks, 32, smem_bytes>>>(d_src, slice, chunks, d_sink);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      GpuTimer timer;
      timer.start();
      kernel_tma_g2s_bandwidth<<<blocks, 32, smem_bytes>>>(d_src, slice, chunks, d_sink);
      timer.stop();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      float ms = timer.elapsedMs();

      bw_gib = bandwidthGiB((double)total, ms * 1e6);
      printf("  TMA G2S bulk-copy BW: %.2f GiB/s (%.1f MB in %.3f ms, %d blocks x %d x %zu KiB)\n",
             bw_gib, (double)total / (1 << 20), ms, blocks, chunks,
             kTmaChunkBytes >> 10);

      CUDA_CHECK(cudaFree(d_sink));
      CUDA_CHECK(cudaFree(d_src));
    } else {
      printf("  Not enough global memory for TMA bandwidth test, skipping.\n");
    }
  }
  g_results.log("tma", "tma_g2s_load_bw_gibps", bw_gib, "GiB/s");
}

} // namespace cmb
