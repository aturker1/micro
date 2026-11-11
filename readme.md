# micro

micro

# Build

Requires Linux x86_64, a host C++ compiler, and a CUDA Toolkit supporting the target GPU.
`bazel.sh` automatically downloads Bazelisk and the pinned Bazel version.

```bash
# B200 (default)
./bazel.sh build //:exp2_bench

# B300
./bazel.sh build --config=b300 //:exp2_bench
```

For the CUDA Toolkit installed on the current devbox:

```bash
./bazel.sh build --config=b300 \
  --repo_env=CUDA_PATH="$HOME/.local/micro-cuda/nvidia/cu13" \
  //:exp2_bench
```

The executable is `bazel-bin/exp2_bench`.
Use `./bazel.sh run` with the same flags to build and run it.

It benchmarks native exp2 and FP32 add/subtract/multiply/FMA using
512 threads and 16 independent chains per thread. The 4096-iteration loop is
unrolled 8× to reduce loop-control overhead.
Output is terminal-only: scalar results per SM clock, plus FLOPs per SM clock
for arithmetic (two FLOPs per FMA).

`src/benchbed.cuh` provides `float benchbed::run(device_function)`:
one callable in, one calls-per-SM-clock value out. The callable is inlined;
its repeated outputs must remain finite. Exp2 and ALU pass their functions from
`src/exp2.cuh` and `src/alu.cuh`. Printing happens only in `src/bench.cu`.

TMEM transfers (B300):

```bash
./bazel.sh run --config=b300 --cuda_archs=sm_103a \
  --repo_env=CUDA_PATH="$HOME/.local/micro-cuda/nvidia/cu13" \
  //:tmem_bench
```

For B200, use `--config=b200 --cuda_archs=sm_100a`.
Measures `32x32b.x1/x8/x16/x32` TMEM loads and stores with one warp.
Each transfer waits for completion; each load supplies the next load's address.
Reports median SM clocks per transfer, including wait and loop overhead.
Allocation and value validation are outside the timing. This measures serialized
completion cost, rather than saturated bandwidth or FA4 kernel runtime.

The same executable also measures a full 128x128 FP32 tile (64 KiB) with
four warps. Each warp transfers its 32 rows using four `x32` instructions,
then waits once. It reports the median of 101 individual tile transfers,
from the earliest thread timestamp to the latest completion timestamp.
The empty timer baseline is printed separately; allocation, initialization,
barriers and validation are outside the timed window. Softmax arithmetic is excluded.
