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
