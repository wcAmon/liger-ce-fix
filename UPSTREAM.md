# Tracking upstream Liger-Kernel

This package copied two files out of [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) **v0.8.0** and rewrote
their cross-entropy math; it has no git ancestry with upstream, so "merging upstream" is a manual port gated by our
tests and benchmark. `upstream.json` is the machine-readable record (base tag, the two files, their sha256 at the base,
the last upstream release a human reviewed, the keywords that mean "upstream may be fixing the same problem").

## What we changed versus v0.8.0

| file (ours) | forked from | our changes |
|---|---|---|
| `liger_ce_fix/ops/cross_entropy.py` | `src/liger_kernel/ops/cross_entropy.py` | pass 1 skips the target column (`lse_ex`), margin `m = x_y − lse_ex`, `loss = softplus(−m)` in log space, `dx_y = −σ(−m)`, `dx_i = e^{x_i−lse_ex}·σ(−m)`; `−inf` guards; dropped bias/weight/softcap/z-loss/label smoothing/return_z_loss |
| `liger_ce_fix/ops/fused_linear_cross_entropy.py` | `src/liger_kernel/ops/fused_linear_cross_entropy.py` | same chunking and grad_h / grad_W accumulation as upstream (weight dtype); calls `ce_fix_kernel`; `needs_grad` from the inputs' `requires_grad`; dropped bias/ce_weight/softcap/z-loss/label smoothing/accuracy/token scaling/accum_dtype |

## Weekly watch (`.github/workflows/upstream-watch.yml`)

Every Monday 01:00 UTC (and on manual dispatch) `tools/upstream_watch.py` asks GitHub for the latest Liger release.
If its tag equals `last_reviewed_tag` nothing happens. Otherwise it diffs the two upstream files between the base tag
and the new release, lists the commits that touched them, greps release notes and commit titles for
`watch_keywords`, and opens one issue labelled `upstream-sync` (or comments on the open one for that tag). It never
opens pull requests. A base-file sha256 mismatch aborts loudly (the base tag's contents must never drift).

## Reviewing a release

1. Read the issue's diff. Classify each hunk: **port** (mechanical, e.g. the grad_W `addmm(out_dtype=…)` rewrite in
   0.8.x) or **re-derive** (anything inside the kernel's two passes — our math replaced them; e.g. upstream's exp2
   trick cannot be pasted in).
2. Apply on a branch; `pytest -q` on a CUDA box must give 70 passed; `python bench/fused_ce_bench.py` must exit 0
   (≥0.95× speed, ≤1.02× memory against the Liger installed in the same venv).
3. Commit, push, tag; in circus bump `tests/test_ce_fix_pin.py` + spec §1.4, reinstall with `--no-deps`, re-run
   `tools.measure_weighted_parity` (the cp5b parity gate pins the kernel hash).
4. Set `last_reviewed_tag` in `upstream.json` to the reviewed tag (also when the decision is "nothing to port") and
   close the issue. The next watch run is then silent until the following release.

`liger-kernel` itself stays a separate dependency of circus (rms_norm, swiglu); upgrading it is a venv change plus the
same circus gates, independent of this package.
