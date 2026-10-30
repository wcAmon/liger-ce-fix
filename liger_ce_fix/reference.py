"""Pure-torch cross-entropy in the paper's log-space form (arXiv 2610.31642, hard targets):

    lse_ex = logsumexp_{j != y} x_j          (target column masked out)
    m      = x_y - lse_ex                     (margin)
    loss   = softplus(-m) = log(1 + e^{-m})   (== lse - x_y in exact arithmetic)

Autograd then yields d loss / d x_y = -sigmoid(-m) and d loss / d x_i = sigmoid(-m) * softmax_ex(x)_i, so the target
gradient stays non-zero until e^{-m} underflows in fp32 (m ~ 87) instead of rounding to 0 past m ~ 16.6.
Any device; always computes in fp32."""
import torch
import torch.nn.functional as F

REDUCTIONS = ("mean", "sum", "none")


def check_reduction(reduction: str) -> None:
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")


def check_target(target: torch.Tensor, n_rows: int, n_classes: int, ignore_index: int) -> None:
    if target.ndim != 1 or target.shape[0] != n_rows:
        raise ValueError(f"target must have shape [{n_rows}], got {tuple(target.shape)}")
    if target.dtype != torch.int64:
        raise ValueError(f"target must be int64, got {target.dtype}")
    kept = target[target != ignore_index]
    if kept.numel() and (kept.min() < 0 or kept.max() >= n_classes):
        raise ValueError(f"target out of range [0, {n_classes})")


def reference_cross_entropy(logits: torch.Tensor, target: torch.Tensor, *, ignore_index: int = -100,
                            reduction: str = "mean") -> torch.Tensor:
    check_reduction(reduction)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [N, V], got {tuple(logits.shape)}")
    n_rows, n_classes = logits.shape
    check_target(target, n_rows, n_classes, ignore_index)
    x = logits.float()
    # a -inf logit would make logsumexp's backward produce nan (exp(-inf - -inf)); the most negative finite value
    # behaves identically in every formula below (its exp is exactly 0) and keeps autograd finite
    x = torch.where(x == float("-inf"), torch.full_like(x, torch.finfo(torch.float32).min / 2), x)
    valid = target != ignore_index
    safe = torch.where(valid, target, torch.zeros_like(target))
    x_y = x.gather(1, safe[:, None]).squeeze(1)
    lse_ex = torch.logsumexp(x.scatter(1, safe[:, None], float("-inf")), dim=1)
    per_token = F.softplus(lse_ex - x_y)                      # softplus(-m); torch's softplus is log-space stable
    per_token = torch.where(valid, per_token, torch.zeros_like(per_token))
    if reduction == "none":
        return per_token
    total = per_token.sum()
    if reduction == "sum":
        return total
    return total / valid.sum().clamp(min=1)                   # all-ignored batch: 0, still connected to logits
