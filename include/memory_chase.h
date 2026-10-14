#pragma once
#include "bench_common.h"
#include <numeric>
#include <random>

namespace cmb {

// Implemented once in memory_chase.cu to avoid duplicate CUDA entry registration.
__global__ void kernel_random_chase_ca(const unsigned int *, int, int, unsigned long long *);
__global__ void kernel_random_chase_cg(const unsigned int *, int, int, unsigned long long *);

template <bool BYPASS_L1>
inline double measureRandomChase(size_t bytes, bool resident) {
  constexpr unsigned int slot_words = 128 / sizeof(unsigned int);
  size_t nodes = bytes / 128;
  std::vector<unsigned int> order(nodes), links(bytes / sizeof(unsigned int), 0);
  for (size_t i = 0; i < links.size(); ++i) {
    unsigned int x = static_cast<unsigned int>(i) + 0x9e3779b9U;
    x = (x ^ (x >> 16)) * 0x85ebca6bU;
    x = (x ^ (x >> 13)) * 0xc2b2ae35U;
    links[i] = x ^ (x >> 16);
  }
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(0xC0DA);
  std::shuffle(order.begin(), order.end(), rng);
  for (size_t i = 0; i < nodes; ++i)
    links[order[i] * slot_words] = order[(i + 1) % nodes] * slot_words;
  unsigned int *d_links;
  unsigned long long *d_out;
  CUDA_CHECK(cudaMalloc(&d_links, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemcpy(d_links, links.data(), bytes, cudaMemcpyHostToDevice));
  // Prefer L1 for .ca probes; no shared memory is used by this kernel.
  if constexpr (BYPASS_L1)
    CUDA_CHECK(cudaFuncSetCacheConfig(kernel_random_chase_cg, cudaFuncCachePreferL1));
  else
    CUDA_CHECK(cudaFuncSetCacheConfig(kernel_random_chase_ca, cudaFuncCachePreferL1));
  const int warmup = resident ? static_cast<int>(nodes * 2) : 1024;
  // Cold samples visit the entire cycle; repeating a short prefix can fit L2.
  const int iters = resident ? (1 << 17) : static_cast<int>(nodes);
  unsigned int expected = 0;
  for (int i = 0; i < warmup + iters; ++i) expected = links[expected];
  std::vector<double> samples;
  for (int sample = 0; sample < 5; ++sample) {
    if constexpr (BYPASS_L1) kernel_random_chase_cg<<<1, 1>>>(d_links, warmup, iters, d_out);
    else kernel_random_chase_ca<<<1, 1>>>(d_links, warmup, iters, d_out);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long out[2];
    CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost));
    if (out[1] != expected) {
      fprintf(stderr, "Pointer chase result mismatch\n");
      exit(EXIT_FAILURE);
    }
    samples.push_back(static_cast<double>(out[0]) / iters);
  }
  CUDA_CHECK(cudaFree(d_links));
  CUDA_CHECK(cudaFree(d_out));
  std::sort(samples.begin(), samples.end());
  return samples[2];
}
} // namespace cmb
