# CUDA / PTX Micro-architecture Benchmark Suite Makefile

CUDA_PATH ?= /usr/local/cuda
NVCC     ?= $(CUDA_PATH)/bin/nvcc

# Use -arch=native (CUDA 11.4+/12+) to target the installed GPU.
# Override with: make ARCH=sm_80
ifndef ARCH
  ARCH_FLAG := -arch=native
else
  ARCH_FLAG := -arch=$(ARCH)
endif

CXXFLAGS ?= -O3 -std=c++17 $(ARCH_FLAG)
INCS     ?= -Iinclude
SRCS      = src/main.cu src/result_logger.cu src/device_info.cu \
            src/topology_extended.cu src/pipeline_depth.cu \
            src/instructions_more.cu src/warp_collectives.cu \
            src/tensor_more.cu src/memory_more.cu src/coissue.cu \
            src/tma.cu src/instruction_throughput.cu \
            src/mem_global.cu src/mem_shared.cu src/mem_cache.cu \
            src/compute_throughput.cu src/compute_latency.cu \
            src/atomics.cu src/tensor_core.cu src/occupancy.cu \
            src/warp_shuffle.cu src/barriers.cu src/instructions_extended.cu \
            src/tensor_core_extended.cu src/pcie.cu src/cluster.cu
OBJS      = $(patsubst src/%.cu, build/%.o, $(SRCS))
TARGET    = cuda_micro_bench

.PHONY: all clean run info

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) $(INCS) $^ -o $@ -lcuda

build/%.o: src/%.cu | build
	$(NVCC) $(CXXFLAGS) $(INCS) -dc $< -o $@

build:
	mkdir -p build

info:
	@echo "NVCC:  $(NVCC)"
	@echo "ARCH:  $(ARCH_FLAG)"
	@echo "SRCS:  $(SRCS)"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf build $(TARGET)
