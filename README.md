# liger-ce-fix

Fused linear cross-entropy (a fork of [Liger Kernel](https://github.com/linkedin/Liger-Kernel) 0.8.0) with the
log-space fix for **softmax collapse** from Barron, Gnanapragasam & Tschannen,
*A Fix for Softmax and Cross-Entropy* ([arXiv 2610.31642](https://arxiv.org/abs/2610.31642), 2026).

## The problem
Hard-target cross-entropy's gradient on the target logit is `p_y − 1`. Once the target leads the rest by a margin
`m = x_y − logsumexp_{j≠y} x_j`, `p_y = 1/(1+e^{−m})` rounds to exactly 1 and that gradient becomes exactly 0, while the
other logits still receive small positive gradients — the gradient no longer sums to zero. The cliff sits at
m ≈ 5.5 nats in bf16, 16.6 in fp32, 36.7 in fp64. Liger computes `loss = lse − x_y` and `dx_y = softmax(x)_y − 1`,
so it collapses at the fp32 cliff even when the inputs are bf16.

## The fix
Compute the margin first and stay in log space:

    lse_ex = logsumexp_{j≠y} x_j        (pass 1 skips the target column)
    m      = x_y − lse_ex
    loss   = softplus(−m)                (== lse − x_y exactly, in real arithmetic)
    dx_y   = −σ(−m)                      dx_i = e^{x_i − lse_ex} · σ(−m)   (i ≠ y)

`e^{−m}` now survives until fp32 underflow (m ≈ 87) instead of vanishing at 16.6; the gradient sums to zero by
construction. Nothing changes for ordinary margins (agreement with `torch.nn.functional.cross_entropy` to 1e-6).

## API
```python
from liger_ce_fix import fused_linear_cross_entropy, cross_entropy, reference_cross_entropy
loss = fused_linear_cross_entropy(hidden, weight, target, ignore_index=-100, reduction="mean")  # CUDA, mean|sum
loss = cross_entropy(logits, target, reduction="none")      # CUDA kernel or CPU reference; overwrites logits on CUDA
loss = reference_cross_entropy(logits, target)              # pure torch, any device
```
Dropped Liger options (`bias`, `ce_weight`, `softcap`, `lse_square_scale`, `label_smoothing`, `return_z_loss`,
`return_token_accuracy`, `return_predicted_tokens`, `accum_dtype`, `use_token_scaling`) raise `ValueError`.
`fused_linear_cross_entropy` does not offer `reduction="none"`: the weight gradient is accumulated inside the forward
pass and cannot be rescaled per row afterwards (Liger has the same limitation, silently).

## Install
    pip install "liger-ce-fix @ git+https://github.com/wcAmon/liger-ce-fix@<commit>"
Requires torch ≥ 2.12; the CUDA path needs triton ≥ 3.7 (`pip install "liger-ce-fix[cuda] @ ..."`).

## Tests and bench
    pytest                                   # CPU tests; GPU tests (marked `cuda`) run when a GPU is visible
    python bench/fused_ce_bench.py           # vs Liger 0.8.0 at 8192×2048×64000 bf16; exit 0 iff ≥0.95× speed, ≤1.02× memory

## License
BSD-2-Clause. Copyright 2024 LinkedIn Corporation (Liger Kernel); modifications Copyright 2026 wcAmon.
