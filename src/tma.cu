#include "bench_common.h"
#include "timer.h"
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace cmb {

static __device__ __forceinline__ unsigned int wallclock() {
  unsigned int r;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(r));
  return r;
}

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
  // keep smem references alive
  if ((int)smem_buf[0] == 0xdeadbeef) out_cycles[1] = 0;
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
    printf("  cuTensorMapEncodeTiled failed: %s\n", errstr ? errstr : "unknown");
    return false;
  }
  return true;
}

void benchmark_tma() {
  printf("=== TMA (cp.async.bulk.tensor) Probes ===\n");

  int cc_major = 0, cc_minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0));
  if (cc_major < 9) {
    printf("  TMA requires SM >= 9.0 (current %d.%d), skipping.\n", cc_major, cc_minor);
    g_results.log("tma", "tma_1d_load_latency_cyc", 0.0, "cycles");
    return;
  }

  CUresult drv_res = cuInit(0);
  if (drv_res != CUDA_SUCCESS) {
    const char *errstr = nullptr;
    cuGetErrorString(drv_res, &errstr);
    printf("  cuInit failed: %s\n", errstr ? errstr : "unknown");
    g_results.log("tma", "tma_1d_load_latency_cyc", 0.0, "cycles");
    return;
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
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("  TMA kernel launch failed: %s\n", cudaGetErrorString(err));
      ok = false;
    } else {
      unsigned int h_cycles;
      CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      latency_cyc = (double)h_cycles / (double)iters;
      printf("  TMA 1D 16-byte load latency: %.2f cycles (%u cycles / %d iters)\n",
             latency_cyc, h_cycles, iters);
    }
  }

  if (!ok) latency_cyc = 0.0;
  g_results.log("tma", "tma_1d_load_latency_cyc", latency_cyc, "cycles");

  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_buf));
}

} // namespace cmb
