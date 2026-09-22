# q3 — Qwen3-32B Q4 GGUF inspector and resident CUDA loader (milestone M0).
#
# Single configuration: optimized, with diagnostics counters enabled so the
# --load-only telemetry is available. No release/diag variants for M0.

CC      ?= cc
CFLAGS  ?= -O3 -g -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -Isrc -pthread
CFLAGS  += -DQ3_DIAGNOSTICS=1

CUDA_HOME ?= $(shell if [ -x /usr/local/cuda/bin/nvcc ]; then \
	printf '%s' /usr/local/cuda; \
	elif command -v nvcc >/dev/null 2>&1; then \
	dirname "$$(dirname "$$(command -v nvcc)")"; \
	else \
	printf '%s' /usr/local/cuda; \
	fi)
NVCC ?= $(CUDA_HOME)/bin/nvcc
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math -Isrc $(NVCC_ARCH_FLAGS)
NVCCFLAGS += -DQ3_DIAGNOSTICS=1
CUDA_LDLIBS ?= -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -Xcompiler -pthread

BIN := q3

C_OBJS := \
	src/q3_gguf.o \
	src/q3_memory.o \
	src/q3_platform.o \
	src/q3_residency.o \
	src/q3_residency_plan.o \
	src/q3_main.o

CUDA_OBJS := \
	cuda/q3_cuda.o \
	cuda/q3_model_loader_cuda.o

OBJS := $(C_OBJS) $(CUDA_OBJS)

TESTS := tests/test_q3_gguf tests/test_q3_residency_plan

.PHONY: all clean test test-gguf test-plan

all: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(CUDA_LDLIBS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

cuda/%.o: cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

tests/test_q3_gguf: tests/test_q3_gguf.c src/q3_gguf.o
	$(CC) $(CFLAGS) -o $@ $< src/q3_gguf.o

tests/test_q3_residency_plan: tests/test_q3_residency_plan.c src/q3_residency_plan.o
	$(CC) $(CFLAGS) -o $@ $< src/q3_residency_plan.o

test-gguf: tests/test_q3_gguf
	mkdir -p tests/fixtures
	./tests/test_q3_gguf

test-plan: tests/test_q3_residency_plan
	./tests/test_q3_residency_plan

test: test-gguf test-plan

clean:
	rm -f $(OBJS) $(BIN) $(TESTS)
