# micro

Linux x86_64 and CUDA required. Build and run (B300):

```bash
./bazel.sh build --config=b300 --cuda_archs=sm_103a \
  --repo_env=CUDA_PATH="$HOME/.local/micro-cuda/nvidia/cu13" \
  //:exp2_bench //:tmem_bench //:mma_bench
bazel-bin/exp2_bench
bazel-bin/tmem_bench
bazel-bin/mma_bench
```

Set `CUDA_PATH` to your toolkit. For B200 use `--config=b200 --cuda_archs=sm_100a`.

## B300 results

B300 SXM6 AC, CUDA 13.2.86, 2026-11-13.

| Operation | Calls / SM clock |
| --- | ---: |
| Exp2 | 31.988 |
| FP32 add | 124.106 |
| FP32 subtract | 124.115 |
| FP32 multiply | 124.123 |
| FP32 FMA | 124.111 |

TMEM clocks per transfer; full tile = 128×128 FP32, four warps.

| Transfer | Load | Store |
| --- | ---: | ---: |
| One warp, x1 | 24.03 | 12.14 |
| One warp, x8 | 27.02 | 15.11 |
| One warp, x16 | 31.01 | 19.08 |
| One warp, x32 | 39.01 | 27.02 |
| Full tile, wait each x32 | 156.00 | 113.00 |
| Full tile, queue four x32 | 93.75 | 80.00 |

FP8 E4M3 MMA → FP32, SMEM operands; M×N×K shape, M×N output.

| Shape | Wait each (clocks) | Queue all (clocks) |
| --- | ---: | ---: |
| 128×128×32 | 425 | 67 |
| 128×128×128 | 582 | 256 |

Wait each completes each transfer/shape; queued mode waits after the batch.
Tile/MMA timings use 256/512-repeat slopes and include waits and loop overhead.
All output checks passed; no spills. FMA counts as two FLOPs per call.
