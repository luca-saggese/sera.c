# q3 — Qwen3-32B Q4 GGUF inspector and resident CUDA loader (milestone M0).
#
# Single configuration: optimized, with diagnostics counters enabled so the
# --load-only telemetry is available. No release/diag variants for M0.

CC      ?= cc
CFLAGS  ?= -O3 -g -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -Isrc -pthread
CFLAGS  += -DQ3_DIAGNOSTICS=1
CFLAGS  += -Isrc/io -Isrc/server

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
NVCCFLAGS += -Isrc/io -Isrc/server
# The native Q4_K MMQ/MMVQ compute closure is C++17 and uses ggml-style
# compatibility headers that live in cuda/mmq/.
NVCC_MMQ_FLAGS := -std=c++17 -Icuda/mmq -Icuda -Xcompiler -Wno-unused-parameter
CUDA_LDLIBS ?= -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -Xcompiler -pthread

BIN := q3

C_OBJS := \
	src/q3_gguf.o \
	src/q3_memory.o \
	src/q3_model.o \
	src/q3_binder.o \
	src/q3_platform.o \
	src/q3_residency.o \
	src/q3_residency_plan.o \
	src/q3_workload.o \
	src/q3_main.o

CUDA_OBJS := \
	src/q3_quant.o \
	cuda/q3_cuda.o \
	cuda/q3_model_loader_cuda.o \
	cuda/q3_cuda_primitives.o \
	cuda/q3_mmq.o \
	cuda/q3_q4_linear.o \
	cuda/q3_forward.o \
	cuda/q3_decide.o \
	cuda/q3_forward_cli.o \
	cuda/q3_bench_q4_linear.o \
	cuda/q3_decide_cli.o

# Native Q4_K MMQ/MMVQ compute closure (COPY -> RENAME -> EDIT from q38.c @ main).
MMQ_OBJS := \
	cuda/mmq/q3_ggml_stubs.o \
	cuda/mmq/quantize.o \
	cuda/mmq/mmvq.o

OBJS := $(C_OBJS) $(CUDA_OBJS) $(MMQ_OBJS)

# M4: the System One HTTP server (docs/M4.md §20). Same compute closure as the
# CLI, plus the HTTP/JSON boundary, the tokenizer and the runtime seam. The CLI
# entry point is excluded: the server owns main().
SERVER_BIN  := build/q3-server
SERVER_OBJS := $(filter-out src/q3_main.o,$(OBJS)) \
	src/io/hd_json.o \
	src/q3_tokenizer.o \
	cuda/q3_systemone.o \
	src/server/q3_server.o

TESTS := tests/test_q3_gguf tests/test_q3_residency_plan tests/test_q3_q4_linear \
	tests/test_q3_forward_primitives tests/test_q3_forward tests/test_q3_decide

.PHONY: all clean test test-gguf test-plan test-q4-linear test-forward-primitives test-forward test-decide server

all: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(CUDA_LDLIBS)

server: $(SERVER_BIN)

$(SERVER_BIN): $(SERVER_OBJS)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ $(SERVER_OBJS) $(CUDA_LDLIBS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

cuda/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_MMQ_FLAGS) -c $< -o $@

cuda/q3_systemone.o: src/q3_systemone.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_MMQ_FLAGS) -c $< -o $@

src/server/%.o: src/server/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

src/io/%.o: src/io/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

cuda/%.o: cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cuda/q3_mmq.o: cuda/q3_mmq.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_MMQ_FLAGS) -c $< -o $@

src/q3_quant.o: src/q3_quant.c
	$(CC) $(CFLAGS) -c $< -o $@

cuda/q3_q4_linear.o: src/q3_q4_linear.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_MMQ_FLAGS) -c $< -o $@

cuda/mmq/%.o: cuda/mmq/%.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_MMQ_FLAGS) -c $< -o $@

tests/test_q3_gguf: tests/test_q3_gguf.c src/q3_gguf.o
	$(CC) $(CFLAGS) -o $@ $< src/q3_gguf.o

tests/test_q3_residency_plan: tests/test_q3_residency_plan.c src/q3_residency_plan.o
	$(CC) $(CFLAGS) -o $@ $< src/q3_residency_plan.o

TEST_Q4_OBJS := $(filter-out src/q3_main.o,$(OBJS))

tests/test_q3_q4_linear: tests/test_q3_q4_linear.cu $(TEST_Q4_OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCC_MMQ_FLAGS) -o $@ $< $(TEST_Q4_OBJS) $(CUDA_LDLIBS)

tests/test_q3_forward_primitives: tests/test_q3_forward_primitives.cu $(TEST_Q4_OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCC_MMQ_FLAGS) -o $@ $< $(TEST_Q4_OBJS) $(CUDA_LDLIBS)

tests/test_q3_forward: tests/test_q3_forward.cu $(TEST_Q4_OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCC_MMQ_FLAGS) -o $@ $< $(TEST_Q4_OBJS) $(CUDA_LDLIBS)

tests/test_q3_decide: tests/test_q3_decide.cu $(TEST_Q4_OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCC_MMQ_FLAGS) -o $@ $< $(TEST_Q4_OBJS) $(CUDA_LDLIBS)

test-gguf: tests/test_q3_gguf
	mkdir -p tests/fixtures
	./tests/test_q3_gguf

test-plan: tests/test_q3_residency_plan
	./tests/test_q3_residency_plan

test-q4-linear: tests/test_q3_q4_linear
	./tests/test_q3_q4_linear

test-forward-primitives: tests/test_q3_forward_primitives
	./tests/test_q3_forward_primitives

test-forward: tests/test_q3_forward
	./tests/test_q3_forward

test-decide: tests/test_q3_decide
	./tests/test_q3_decide

test: test-gguf test-plan test-q4-linear test-forward-primitives test-forward test-decide

clean:
	rm -f $(OBJS) $(BIN) $(TESTS) $(SERVER_BIN)
	rm -rf src/server/*.o src/io/*.o build
