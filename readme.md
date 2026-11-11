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

It benchmarks native exp2 and FP32 add, subtract, multiply, and FMA using
512 threads and 16 independent chains per thread.
Output is terminal-only: scalar results per SM clock, plus FLOPs per SM clock
for arithmetic (two FLOPs per FMA).
