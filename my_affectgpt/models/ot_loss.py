"""
Optimal Transport (Sinkhorn) Loss for Multi-modal Emotion Distillation.

Key design decisions (addressing known distillation risks):
  1. Cosine cost: bounded [0, 2], avoids epsilon/cost scale mismatch
  2. Frozen teacher projector: prevents "shortcut" where teacher_proj adapts
     to make OT loss small without the student truly learning teacher geometry
  3. Float32 throughout: prevents bf16/fp16 precision issues in logsumexp/exp
  4. Debiased Sinkhorn divergence: eliminates entropic regularization bias
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cost_matrix_cosine(x, y):
    """
    Cosine distance cost matrix: C_ij = 1 - cos(x_i, y_j).
    Range: [0, 2], numerically well-behaved with epsilon ~ 0.1.
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    cos_sim = torch.bmm(x_norm, y_norm.transpose(1, 2))
    return 1.0 - cos_sim


def sinkhorn_divergence(x, y, epsilon=0.1, num_iters=20, mask_x=None, mask_y=None):
    """
    Debiased Sinkhorn divergence between two point clouds.

    S(x,y) = OT_eps(x,y) - 0.5*OT_eps(x,x) - 0.5*OT_eps(y,y)

    All computation forced to float32 for numerical stability.

    Args:
        x: [batch, n, d] teacher projected features
        y: [batch, m, d] student projected features
        epsilon: entropic regularization (should be ~0.1 for cosine cost in [0,2])
        num_iters: Sinkhorn iterations
        mask_x: [batch, n] boolean mask for valid positions
        mask_y: [batch, m] boolean mask for valid positions
    Returns:
        loss: scalar, non-negative Sinkhorn divergence
    """
    # [Implementation removed - paper under review]
    raise NotImplementedError("Contact authors for details")

def _sinkhorn_cost(x, y, epsilon, num_iters, mask_x=None, mask_y=None):
    """
    Entropic OT cost via log-domain Sinkhorn iterations with cosine cost.
    """
    C = _cost_matrix_cosine(x, y)  # [batch, n, m], in [0, 2]
    batch_size, n, m = C.shape
    device = C.device

    # Marginal distributions (uniform over valid positions, zero on masked)
    if mask_x is not None:
        log_mu = torch.where(
            mask_x,
            torch.zeros((), device=device, dtype=x.dtype),
            torch.tensor(-1e9, device=device, dtype=x.dtype))
        log_mu = log_mu - torch.logsumexp(log_mu, dim=1, keepdim=True)
    else:
        log_mu = torch.full((batch_size, n), -math.log(n),
                            device=device, dtype=x.dtype)

    if mask_y is not None:
        log_nu = torch.where(
            mask_y,
            torch.zeros((), device=device, dtype=y.dtype),
            torch.tensor(-1e9, device=device, dtype=y.dtype))
        log_nu = log_nu - torch.logsumexp(log_nu, dim=1, keepdim=True)
    else:
        log_nu = torch.full((batch_size, m), -math.log(m),
                            device=device, dtype=y.dtype)

    log_K = -C / epsilon  # with cosine cost [0,2] and eps=0.1: log_K in [-20, 0]

    u = torch.zeros(batch_size, n, device=device, dtype=x.dtype)
    v = torch.zeros(batch_size, m, device=device, dtype=y.dtype)

    for _ in range(num_iters):
        u = log_mu - torch.logsumexp(log_K + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(log_K + u.unsqueeze(2), dim=1)

    log_T = u.unsqueeze(2) + log_K + v.unsqueeze(1)
    T = torch.exp(log_T)
    cost = (T * C).sum(dim=(1, 2))

    return cost


class OTProjector(nn.Module):
    """
    Learnable projection heads for OT alignment.

    CRITICAL: teacher_proj is FROZEN with orthogonal initialization.
    This defines a fixed reference geometry in the common space,
    preventing the "shortcut" where teacher_proj adapts to make
    OT loss small without the student truly learning teacher structure.

    Only student_proj is trained — the student must learn to map
    its representations into the teacher's fixed reference frame.
    """
    def __init__(self, teacher_dim, student_dim, common_dim=256):
        super().__init__()
        self.teacher_proj = nn.Sequential(
            nn.Linear(teacher_dim, common_dim, bias=False),
            nn.LayerNorm(common_dim),
        )
        self.student_proj = nn.Sequential(
            nn.Linear(student_dim, common_dim),
            nn.LayerNorm(common_dim),
        )
        # Orthogonal init for teacher: preserves distances in the subspace
        nn.init.orthogonal_(self.teacher_proj[0].weight)

        # Freeze teacher projector
        for p in self.teacher_proj.parameters():
            p.requires_grad = False

    def forward(self, teacher_feats, student_feats):
        """
        Args:
            teacher_feats: [batch, n, teacher_dim]
            student_feats: [batch, m, student_dim]
        Returns:
            proj_teacher: [batch, n, common_dim] (no grad w.r.t. teacher_proj weights)
            proj_student: [batch, m, common_dim] (has grad w.r.t. student_proj weights)
        """
        # [Implementation removed - paper under review]
        raise NotImplementedError("Contact authors for details")
