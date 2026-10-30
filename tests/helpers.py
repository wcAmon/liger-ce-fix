import math
import torch


def margin_logits(V, margin, target, rows=1, device="cpu", dtype=torch.float32):
    """Rows whose non-target logits are all 0, so lse_ex = log(V-1) exactly and x_y - lse_ex = margin."""
    x = torch.zeros(rows, V, device=device, dtype=torch.float32)
    x[:, target] = margin + math.log(V - 1)
    y = torch.full((rows,), target, device=device, dtype=torch.int64)
    return x.to(dtype), y
