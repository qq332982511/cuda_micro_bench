#include "bench_common.h"

namespace cmb {
extern __shared__ char smem_raw[];
__global__ void kernel_shared_bw(float *out, int iters) {
  float *smem = reinterpret_cast<float *>(smem_raw);
  int tid = threadIdx.x;
  for (int i = tid; i < 4096; i += blockDim.x) smem[i] = static_cast<float>(i);
  __syncthreads();
  unsigned int base = __cvta_generic_to_shared(smem);
  float x=0,y=0,z=0,w=0;
#pragma unroll 16
  for (int r = 0; r < iters; ++r) {
    unsigned int addr = base + ((tid * 4 + r * 1024) & 4095) * 4;
    asm volatile("ld.volatile.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(x),"=f"(y),"=f"(z),"=f"(w) : "r"(addr) : "memory");
  }
  out[blockIdx.x * blockDim.x + tid] = x+y+z+w;
}
// Every word points to itself. Addresses depend on the preceding load; one
// warp isolates load-to-use cost from the interference of other resident warps.
__global__ void kernel_shared_chase(unsigned int *out, int stride, int iters) {
  unsigned int *smem = reinterpret_cast<unsigned int *>(smem_raw);
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) smem[i] = i * 4;
  __syncthreads();
  unsigned int base = __cvta_generic_to_shared(smem);
  unsigned int idx = threadIdx.x * stride * 4;
  unsigned int t0 = clockVal();
#pragma unroll 32
  for (int r = 0; r < iters; ++r)
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(idx) : "r"(base+idx) : "memory");
  unsigned int t1 = clockAfter(idx);
  out[threadIdx.x] = t1-t0;
}
void benchmark_shared_memory() {
  DeviceInfo info=queryDevice(0);
  printf("=== Shared Memory Benchmarks ===\n");
  const int iters=1<<14, threads=256, blocks=info.sm_count*4;
  float *out;
  CUDA_CHECK(cudaMalloc(&out,blocks*threads*sizeof(float)));
  std::vector<double> samples;
  for(int i=0;i<6;++i) {
    GpuTimer timer; timer.start();
    kernel_shared_bw<<<blocks,threads,16<<10>>>(out,iters);
    CUDA_CHECK(cudaGetLastError()); timer.stop();
    double bw=bandwidthGiB(static_cast<double>(blocks)*threads*iters*16,timer.elapsedMs()*1e6);
    if(i) samples.push_back(bw);
  }
  std::sort(samples.begin(),samples.end());
  printf("  Shared vector load BW: %.2f GiB/s\n",samples[2]);
  g_results.log("memory_shared","bandwidth_gibps",samples[2],"GiB/s");
  CUDA_CHECK(cudaFree(out));
  unsigned int *cycles;
  CUDA_CHECK(cudaMalloc(&cycles,32*sizeof(unsigned int)));
  auto measure=[&](int stride,int lanes) {
    std::vector<double> times;
    for(int sample=0;sample<5;++sample) {
      kernel_shared_chase<<<1,lanes,4096>>>(cycles,stride,iters);
      CUDA_CHECK(cudaGetLastError());
      unsigned int h;
      CUDA_CHECK(cudaMemcpy(&h,cycles,sizeof(h),cudaMemcpyDeviceToHost));
      times.push_back(static_cast<double>(h)/iters);
    }
    std::sort(times.begin(),times.end());return times[2];
  };
  double latency=measure(0,32);
  g_results.log("memory_shared","latency_cycles",latency,"cycles");
  g_results.log("memory","shared_memory_latency_cyc",latency,"cycles");
  printf("  Shared dependent broadcast load latency: %.2f cycles\n",latency);
  for(int stride=1;stride<=32;++stride) {
    double cyc=measure(stride,32);
    printf("    stride %2d: %.2f cycles/load (one warp)\n",stride,cyc);
    if(stride==1||stride==2||stride==16||stride==17||stride==32) {
      char key[64];snprintf(key,sizeof(key),"bank_conflict_stride_%d_cycles",stride);
      g_results.log("memory_shared",key,cyc,"cycles/warp-load");
    }
  }
  CUDA_CHECK(cudaFree(cycles));
}
} // namespace cmb
