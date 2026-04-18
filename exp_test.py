"""
Microbenchmark: GPU exp() throughput, latency, and ops/clock/SM, written in CuTe DSL.

Strategy
--------
- Launch 1 block per SM (saturates the device but keeps occupancy controlled).
- Each thread holds K independent register values and runs OUTER iterations of
  `v_i = exp(v_i * scale)`. This produces K independent dependency chains so the
  SFU pipeline can stay full and we measure throughput, not pipeline latency.
- A second config (K=1) measures *latency* of one exp (back-to-back dependent ops).
- A small per-iter `*scale` keeps values bounded so we never NaN/Inf.
- Inputs/outputs touched once, so the kernel is SFU-bound, not memory-bound.
- We time with CUDA events.

Reported numbers
----------------
- Wall: total exp ops / sec (Gops/s).
- Per-SM per-clock: total_ops / (time * num_SMs * clock_hz). Compare to the SFU
  peak for your arch (Hopper/Blackwell SFU EX2: 16 ops/clk/SM).
- Latency: time per op along a single dependency chain (clocks).
"""

import argparse

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@cute.kernel
def _exp_throughput_kernel(
    gIn: cute.Tensor,
    gOut: cute.Tensor,
    OUTER: cutlass.Constexpr,
    K: cutlass.Constexpr,
    FAST: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gidx = bid * bdim + tid
    base = gidx * K

    # K independent dependency chains, kept entirely in registers.
    vs = [None] * K
    for i in cutlass.range_constexpr(K):
        vs[i] = gIn[base + i]

    scale = gIn.element_type(0.001)
    # Outer loop is dynamic (small IR), inner K is unrolled at trace time.
    for _ in cutlass.range(OUTER):
        for i in cutlass.range_constexpr(K):
            vs[i] = cute.math.exp(vs[i] * scale, fastmath=FAST)

    s = vs[0]
    for i in cutlass.range_constexpr(1, K):
        s = s + vs[i]
    gOut[gidx] = s


@cute.jit
def _exp_throughput_launch(
    x: cute.Tensor,
    y: cute.Tensor,
    OUTER: cutlass.Constexpr,
    K: cutlass.Constexpr,
    GRID: cutlass.Constexpr,
    BLOCK: cutlass.Constexpr,
    FAST: cutlass.Constexpr,
):
    _exp_throughput_kernel(x, y, OUTER, K, FAST).launch(
        grid=[GRID, 1, 1], block=[BLOCK, 1, 1]
    )
    return y


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_kernel_cache: dict[tuple, object] = {}

_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _get_kernel(outer: int, k: int, grid: int, block: int, fast: bool, dtype: torch.dtype):
    key = (outer, k, grid, block, fast, dtype)
    fn = _kernel_cache.get(key)
    if fn is not None:
        return fn

    n_threads = grid * block
    x = torch.empty(n_threads * k, dtype=dtype, device="cuda").uniform_(-0.5, 0.5)
    y = torch.empty(n_threads, dtype=dtype, device="cuda")
    fn = cute.compile(
        _exp_throughput_launch,
        from_dlpack(x, assumed_align=16),
        from_dlpack(y, assumed_align=16),
        outer, k, grid, block, fast,
    )
    _kernel_cache[key] = fn
    return fn


def bench(outer: int, k: int, grid: int, block: int, fast: bool, dtype: torch.dtype, iters: int = 50, warmup: int = 5):
    fn = _get_kernel(outer, k, grid, block, fast, dtype)

    n_threads = grid * block
    x = torch.empty(n_threads * k, dtype=dtype, device="cuda").uniform_(-0.5, 0.5)
    y = torch.empty(n_threads, dtype=dtype, device="cuda")
    x_c = from_dlpack(x, assumed_align=16)
    y_c = from_dlpack(y, assumed_align=16)

    for _ in range(warmup):
        fn(x_c, y_c)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x_c, y_c)
    end.record()
    torch.cuda.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000.0 / iters
    total_ops = n_threads * outer * k
    return elapsed_s, total_ops, y


def _gpu_props():
    p = torch.cuda.get_device_properties(0)
    name = p.name
    n_sm = p.multi_processor_count
    # Use sustained boost clock; nvidia-smi shows max SM clock
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            text=True,
        )
        clock_mhz = float(out.strip().splitlines()[0])
    except Exception:
        clock_mhz = float(p.clock_rate / 1000)  # kHz -> MHz
    return name, n_sm, clock_mhz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", type=int, default=4096, help="dynamic outer loop count")
    ap.add_argument("--k", type=int, default=8, help="independent dependency chains per thread")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--blocks-per-sm", type=int, default=1)
    ap.add_argument("--no-fast", action="store_true", help="disable fastmath (use libdevice exp, not MUFU.EX2)")
    ap.add_argument("--dtype", choices=list(_DTYPES), default="fp32")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    name, n_sm, clk_mhz = _gpu_props()
    grid = n_sm * args.blocks_per_sm
    fast = not args.no_fast
    dtype = _DTYPES[args.dtype]

    print(f"GPU: {name}  SMs={n_sm}  clk_max={clk_mhz:.0f} MHz")
    print(f"grid={grid} block={args.block} K={args.k} OUTER={args.outer} fastmath={fast} dtype={args.dtype}")
    print()

    # Throughput run
    t_s, ops, _ = bench(args.outer, args.k, grid, args.block, fast, dtype, iters=args.iters)
    gops = ops / t_s / 1e9
    ops_per_clk_per_sm = ops / (t_s * (clk_mhz * 1e6) * n_sm)
    print(f"[throughput, K={args.k}]")
    print(f"  time/iter      : {t_s*1e3:8.3f} ms")
    print(f"  total ops/iter : {ops:,}")
    print(f"  throughput     : {gops:8.1f} Gops/s")
    print(f"  per SM         : {gops/n_sm*1000:8.2f} Mops/s/SM")
    print(f"  ops/clk/SM     : {ops_per_clk_per_sm:8.2f}   (B200 SFU peak EX2 ~= 16)")
    print()

    # Latency run: K=1 means a single dependency chain — each exp must wait
    # for the previous one (plus the *scale FMA), so we get steady-state issue
    # latency along the chain.
    t_s_l, _, _ = bench(args.outer, 1, grid, args.block, fast, dtype, iters=args.iters)
    # per-thread chain length = OUTER ops; time per op along the chain:
    sec_per_op = t_s_l / args.outer
    clks_per_op = sec_per_op * clk_mhz * 1e6
    print(f"[latency, K=1]")
    print(f"  time/iter      : {t_s_l*1e3:8.3f} ms")
    print(f"  per-op latency : {clks_per_op:8.2f} clocks  ({sec_per_op*1e9:.2f} ns)")


if __name__ == "__main__":
    main()
