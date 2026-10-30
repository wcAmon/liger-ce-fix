"""Forked from liger-kernel 0.8.0, liger_kernel/ops/cross_entropy.py (BSD-2-Clause, Copyright 2024 LinkedIn Corporation).

Changes versus Liger:
  * pass 1 (online max/sum) skips the target column -> lse_ex; margin m = x_y - lse_ex
  * loss = softplus(-m) in a piecewise log-space form (libdevice log1p); never lse - x_y
  * grad: dx_i = exp(x_i - lse_ex) * sigmoid(-m) for i != y, dx_y = -sigmoid(-m); sigmoid via exp(-|m|)
  * removed: class weights, softcapping, z-loss, label smoothing, token accuracy, predicted tokens
The row loop, in-place gradient write-back and mean scaling inside the kernel are Liger's."""
import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import log1p as _log1p
except ModuleNotFoundError:  # NGC containers
    from triton.language.extra.cuda.libdevice import log1p as _log1p

from liger_ce_fix.reference import check_reduction, check_target

MAX_FUSED_SIZE = 65536 // 2
NUM_WARPS = 32


@triton.jit
def ce_fix_kernel(
    X_ptr, X_stride, Y_ptr, Y_stride, loss_ptr, loss_stride, n_cols, n_non_ignore, ignore_index,
    reduction: tl.constexpr, BLOCK_SIZE: tl.constexpr, HAS_GRADIENTS: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    X_ptr += program_id * X_stride
    if y == ignore_index:
        if HAS_GRADIENTS:
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return
    loss_ptr += program_id * loss_stride
    x_y = tl.load(X_ptr + y).cast(tl.float32)

    # pass 1: online logsumexp over every column except the target
    m_ex = float("-inf")
    d_ex = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=(X_offsets < n_cols) & (X_offsets != y),
                          other=float("-inf")).cast(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m_ex, block_max)
        # all columns so far -inf (V == 1, or every non-target logit is -inf): keep exp(-inf - m_safe) = 0, not nan
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        d_ex = d_ex * tl.exp(m_ex - m_safe) + tl.sum(tl.exp(X_block - m_safe))
        m_ex = m_new
    lse_ex = m_ex + tl.log(d_ex)                      # -inf when nothing survived: margin = +inf, s = 0, loss = 0
    lse_safe = tl.where(lse_ex == float("-inf"), 0.0, lse_ex)
    margin = x_y - lse_ex

    # s = sigmoid(-margin) without overflow: e = exp(-|margin|)
    e_abs = tl.exp(-tl.abs(margin))
    s = tl.where(margin > 0, e_abs / (1.0 + e_abs), 1.0 / (1.0 + e_abs))

    # loss = softplus(-margin), piecewise in log space
    z = -margin
    loss = tl.where(z > 20.0, z, tl.where(z < -20.0, tl.exp(z), _log1p(tl.exp(tl.minimum(z, 20.0)))))

    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).cast(tl.float32)
            g = tl.exp(X_block - lse_safe) * s
            g = tl.where(X_offsets != y, g, -s)
            if reduction == "mean":
                g = g / n_non_ignore
            tl.store(X_ptr + X_offsets, g, mask=X_offsets < n_cols)

    tl.debug_barrier()
    if reduction == "mean":
        loss = loss / n_non_ignore
    tl.store(loss_ptr, loss)


def cross_entropy_forward(logits, target, ignore_index, reduction):
    """Runs the kernel over [N, V] logits. Returns (loss, grad_logits); grad_logits IS `logits` (contiguous copy if
    needed). When logits requires grad it is overwritten in place with d loss / d logits (ignored rows zeroed);
    otherwise the kernel only reads it and the returned tensor still holds the logits."""
    check_reduction(reduction)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [N, V], got {tuple(logits.shape)}")
    n_rows, n_cols = logits.shape
    check_target(target, n_rows, n_cols, ignore_index)
    # read before any copy: inside autograd.Function.forward grad mode is off, so a .contiguous() copy of a tensor
    # that requires grad comes back with requires_grad=False and would silently skip the gradient pass
    has_gradients = logits.requires_grad
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    target = target.contiguous()
    n_non_ignore = int((target != ignore_index).sum().item())
    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=logits.device)
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    if n_rows:
        ce_fix_kernel[(n_rows,)](
            X_ptr=logits, X_stride=logits.stride(-2), Y_ptr=target, Y_stride=target.stride(-1),
            loss_ptr=loss_1d, loss_stride=loss_1d.stride(-1), n_cols=n_cols, n_non_ignore=max(n_non_ignore, 1),
            ignore_index=ignore_index, reduction=reduction, BLOCK_SIZE=BLOCK_SIZE,
            HAS_GRADIENTS=has_gradients, num_warps=NUM_WARPS,
        )
    loss = loss_1d if reduction == "none" else loss_1d.sum()
    return loss, logits


def scale_by_grad_output(grad, grad_output):
    """In place: grad *= grad_output (scalar, or one value per row when grad_output is 1-D)."""
    if grad_output.ndim == 0:
        if grad_output.item() != 1.0:
            grad.mul_(grad_output.to(grad.dtype))
    else:
        grad.mul_(grad_output.to(grad.dtype).unsqueeze(1))
    return grad


class CrossEntropyFunction(torch.autograd.Function):
    """apply(logits, target, ignore_index, reduction). The gradient is written INTO `logits` during forward (Liger's
    design: the logits buffer is dead after the loss anyway), so callers must not reuse `logits` afterwards."""

    @staticmethod
    def forward(ctx, logits, target, ignore_index, reduction):
        loss, grad = cross_entropy_forward(logits, target, ignore_index, reduction)
        ctx.save_for_backward(grad.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return scale_by_grad_output(grad, grad_output), None, None, None


def cross_entropy_autograd(logits, target, ignore_index, reduction):
    """CUDA entry behind liger_ce_fix.cross_entropy. Function.forward cannot see grad mode, so the decision is made
    here: with grad mode on and logits requiring grad the autograd Function runs (and overwrites logits with the
    gradient); otherwise a plain forward on detached logits runs the kernel read-only and leaves logits intact."""
    if torch.is_grad_enabled() and logits.requires_grad:
        return CrossEntropyFunction.apply(logits, target, ignore_index, reduction)
    loss, _ = cross_entropy_forward(logits.detach(), target, ignore_index, reduction)
    return loss
