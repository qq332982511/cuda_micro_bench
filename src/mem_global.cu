#include "memory_chase.h"

namespace cmb {
// Fill with incompressible data so bandwidth is not an all-zero fast path.
__global__ void kernel_init_stream(unsigned int *data, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    unsigned int x = static_cast<unsigned int>(i) + 0x9e3779b9U;
    x = (x ^ (x >> 16)) * 0x85ebca6bU;
    x = (x ^ (x >> 13)) * 0xc2b2ae35U;
    data[i] = x ^ (x >> 16);
  }
}
__global__ void kernel_stream_copy(const uint4 *src, uint4 *dst, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    dst[i] = src[i];
}
__global__ void kernel_stream_read(const uint4 *src, unsigned int *out, int n) {
  unsigned int a=0,b=0,c=0,d=0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    unsigned int x,y,z,w;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(x),"=r"(y),"=r"(z),"=r"(w) : "l"(src+i) : "memory");
    a^=x; b^=y; c^=z; d^=w;
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a^b^c^d;
}
void benchmark_global_memory() {
  DeviceInfo info = queryDevice(0);
  printf("=== Global Memory Benchmarks (median of 5) ===\n");
  const size_t sizes[] = {16ULL << 20, 64ULL << 20, 128ULL << 20, 256ULL << 20};
  double reported_read=0,reported_copy=0;
  size_t reported_bytes=0;
  int blocks=info.sm_count*8, threads=256;
  for(size_t bytes:sizes) {
    if(bytes>info.total_global_mem/4) continue;
    uint4 *a,*b;
    unsigned int *sink;
    CUDA_CHECK(cudaMalloc(&a,bytes)); CUDA_CHECK(cudaMalloc(&b,bytes));
    CUDA_CHECK(cudaMalloc(&sink,blocks*threads*sizeof(unsigned int)));
    kernel_init_stream<<<blocks,threads>>>(reinterpret_cast<unsigned int *>(a),bytes/4);
    CUDA_CHECK(cudaGetLastError());
    auto measure=[&](bool copy) {
      std::vector<double> samples;
      for(int r=0;r<6;r++) {
        GpuTimer timer; timer.start();
        if(copy) kernel_stream_copy<<<blocks,threads>>>(a,b,bytes/16);
        else kernel_stream_read<<<blocks,threads>>>(a,sink,bytes/16);
        CUDA_CHECK(cudaGetLastError()); timer.stop();
        double ms=timer.elapsedMs();
        if(r) samples.push_back(bandwidthGiB(bytes*(copy?2.0:1.0),ms*1e6));
      }
      std::sort(samples.begin(),samples.end()); return samples[2];
    };
    double copy_bw=measure(true),read_bw=measure(false);
    printf("  %4zu MiB: copy %.2f GiB/s (read+write), read %.2f GiB/s\n",bytes>>20,copy_bw,read_bw);
    reported_bytes=bytes; reported_copy=copy_bw; reported_read=read_bw;
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(sink));
  }
  g_results.log("memory_dram","read_bw_gibps",reported_read,"GiB/s");
  g_results.log("memory_dram","copy_max_bw_gibps",reported_copy,"GiB/s");
  g_results.log("memory","dram_read_bw_gbps",reported_read*1.073741824,"GB/s");
  g_results.log("memory_dram","stream_working_set_bytes",reported_bytes,"bytes");
  // Visit every 128-byte slot. Eight times L2 also exceeds L2 capacity
  // when only the 32-byte sector containing each link is fetched.
  size_t bytes=std::max<size_t>(256ULL<<20,static_cast<size_t>(info.l2_cache_size)*8);
  bytes=std::min(bytes,info.total_global_mem/8);
  double cyc=measureRandomChase<true>(bytes,false);
  double ns=cyclesToNominalNs(cyc,info.clock_rate_khz);
  printf("  DRAM random chase (.cg, %zu MiB): %.2f cycles (%.2f nominal ns)\n",bytes>>20,cyc,ns);
  g_results.log("memory","dram_latency_cyc",cyc,"cycles");
  g_results.log("memory_dram","pointer_chase_latency_ns",ns,"nominal ns");
  g_results.log("memory_dram","chase_working_set_bytes",bytes,"bytes");
  g_results.log("memory_dram","chase_loads_per_sample",bytes/128,"loads");
  g_results.log("memory_dram","chase_min_sector_footprint_bytes",bytes/4,"bytes");
}
} // namespace cmb
