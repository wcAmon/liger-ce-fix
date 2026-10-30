"""Forward+backward throughput and peak memory of liger_ce_fix.fused_linear_cross_entropy vs Liger 0.8.0's
liger_fused_linear_cross_entropy at the circus shape (8192 tokens x 2048 hidden, 64000 vocab, bf16).
Exit 0 when ours is >= 0.95x Liger's speed and <= 1.02x its peak memory (spec 2026-10-30-ce-fix-design §3.7).
Requires CUDA and liger-kernel installed (the circus venv has both)."""
import argparse
import json
import time
from pathlib import Path

import torch

MIN_SPEED_RATIO = 0.95
MAX_MEM_RATIO = 1.02


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--vocab", type=int, default=64000)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def verdict(ours, liger, min_speed_ratio=MIN_SPEED_RATIO, max_mem_ratio=MAX_MEM_RATIO):
    speed = liger["ms_per_iter"] / ours["ms_per_iter"]          # >1 means ours is faster
    mem = ours["peak_bytes"] / liger["peak_bytes"]
    ok = speed >= min_speed_ratio and mem <= max_mem_ratio
    return ok, f"speed {speed:.3f}x (min {min_speed_ratio}), memory {mem:.3f}x (max {max_mem_ratio})"


def _time(fn, h, w, y, iters, warmup):
    for _ in range(warmup):
        loss = fn(h, w, y)
        loss.backward()
        h.grad = None; w.grad = None
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        loss = fn(h, w, y)
        loss.backward()
        h.grad = None; w.grad = None
    torch.cuda.synchronize()
    return dict(ms_per_iter=(time.perf_counter() - start) * 1000 / iters,
                peak_bytes=torch.cuda.max_memory_allocated(), loss=float(loss))


def main(argv=None):
    a = parse_args(argv)
    from liger_ce_fix import fused_linear_cross_entropy as ours_fn
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy as liger_fn
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    torch.manual_seed(0)
    h = (torch.randn(a.tokens, a.hidden, device="cuda") * 0.5).to(dtype).requires_grad_(True)
    w = (torch.randn(a.vocab, a.hidden, device="cuda") * 0.02).to(dtype).requires_grad_(True)
    y = torch.randint(0, a.vocab, (a.tokens,), device="cuda")
    y[::50] = -100
    liger = _time(lambda h, w, y: liger_fn(h, w, y, ignore_index=-100), h, w, y, a.iters, a.warmup)
    ours = _time(lambda h, w, y: ours_fn(h, w, y, ignore_index=-100), h, w, y, a.iters, a.warmup)
    ok, why = verdict(ours, liger)
    import liger_ce_fix, liger_kernel
    result = dict(shape=dict(tokens=a.tokens, hidden=a.hidden, vocab=a.vocab, dtype=a.dtype, iters=a.iters),
                  ours=ours, liger=liger, verdict=why, pass_=ok, device=torch.cuda.get_device_name(),
                  torch=torch.__version__, liger_ce_fix=dict(version=liger_ce_fix.__version__, commit=liger_ce_fix.__commit__),
                  liger_kernel=getattr(liger_kernel, "__version__", "unknown"))
    print(json.dumps(result, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("x") as f:
            json.dump(result, f, indent=1)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
