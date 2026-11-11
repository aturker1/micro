NVCC ?= nvcc
CUOBJDUMP ?= cuobjdump
ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xptxas=-v,-warn-spills

TARGET := build/exp2_bench

.PHONY: all run sass

all: $(TARGET)

# bench.cu includes the templated kernel; compile only this translation unit.
$(TARGET): src/bench.cu src/exp2.cuh | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

build results:
	mkdir -p $@

run: $(TARGET) | results
	cd results && ../$(TARGET) > exp2_run.txt 2>&1 || \
		{ cat results/exp2_run.txt >&2; exit 1; }
	cat results/exp2_run.txt

sass: $(TARGET) | results
	$(CUOBJDUMP) --dump-sass $(TARGET) > results/exp2.sass
