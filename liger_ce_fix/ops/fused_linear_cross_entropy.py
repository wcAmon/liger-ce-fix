"""Forked from liger-kernel 0.8.0, liger_kernel/ops/fused_linear_cross_entropy.py (BSD-2-Clause, Copyright 2024 LinkedIn
Corporation). Same chunking (chunk_size = next_pow2(ceil(BT / ceil(V/H)))), same grad_h = dlogits @ W and
grad_W += dlogits^T @ h accumulation in the weight dtype, same in-forward gradient computation. Removed bias,
ce_weight, softcap, z-loss, label smoothing, accuracy, token scaling, accum_dtype. The kernel is ce_fix_kernel."""
import torch
import triton

from liger_ce_fix.ops.cross_entropy import MAX_FUSED_SIZE, NUM_WARPS, ce_fix_kernel, scale_by_grad_output
from liger_ce_fix.reference import check_reduction, check_target

FUSED_REDUCTIONS = ("mean", "sum")


def fused_linear_cross_entropy_forward(hidden, weight, target, ignore_index, reduction):
    check_reduction(reduction)
    if reduction not in FUSED_REDUCTIONS:
        raise ValueError("fused_linear_cross_entropy supports reduction 'mean' or 'sum' only: with 'none' the weight "
                         "gradient cannot be rescaled per row afterwards; use cross_entropy on materialized logits")
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError(f"hidden [N, H] and weight [V, H] disagree: {tuple(hidden.shape)} vs {tuple(weight.shape)}")
    BT, H = hidden.shape
    V = weight.shape[0]
    check_target(target, BT, V, ignore_index)
    device = hidden.device
    # autograd.Function.forward runs with grad mode OFF, so torch.is_grad_enabled() is useless here; the inputs'
    # requires_grad flags are what tells us whether a backward can follow (Liger decides the same way)
    needs_grad = hidden.requires_grad or weight.requires_grad
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)
    grad_hidden = torch.zeros_like(hidden) if hidden.requires_grad else None
    grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    target = target.contiguous()
    n_non_ignore = int((target != ignore_index).sum().item())
    for chunk_id in range(num_chunks):
        start, end = chunk_id * chunk_size, min((chunk_id + 1) * chunk_size, BT)
        h_chunk = hidden[start:end]
        logits_chunk = (h_chunk @ weight.t()).contiguous()          # [rows, V] in the input dtype
        t_chunk = target[start:end]
        loss_slice = loss_1d[start:end]
        ce_fix_kernel[(end - start,)](
            X_ptr=logits_chunk, X_stride=logits_chunk.stride(-2), Y_ptr=t_chunk, Y_stride=t_chunk.stride(-1),
            loss_ptr=loss_slice, loss_stride=loss_slice.stride(-1), n_cols=V, n_non_ignore=max(n_non_ignore, 1),
            ignore_index=ignore_index, reduction=reduction, BLOCK_SIZE=BLOCK_SIZE, HAS_GRADIENTS=needs_grad,
            num_warps=NUM_WARPS,
        )
        if needs_grad:                                                # logits_chunk now holds dloss/dlogits
            if grad_hidden is not None:
                grad_hidden[start:end] = logits_chunk @ weight
            if grad_weight is not None:
                grad_weight += torch.mm(logits_chunk.t(), h_chunk).float()
    return loss_1d.sum(), grad_hidden, grad_weight


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    """apply(hidden, weight, target, ignore_index, reduction) -> scalar loss. Gradients are computed in forward and
    scaled by grad_output in backward (Liger's design: no logits kept alive)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden, weight, target, ignore_index, reduction):
        loss, grad_hidden, grad_weight = fused_linear_cross_entropy_forward(hidden, weight, target, ignore_index, reduction)
        ctx.save_for_backward(grad_hidden, grad_weight)
        return loss

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        if grad_hidden is not None:
            grad_hidden = scale_by_grad_output(grad_hidden, grad_output)
        if grad_weight is not None:
            grad_weight = scale_by_grad_output(grad_weight, grad_output)
        return grad_hidden, grad_weight, None, None, None
