# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor


def _default_skeleton_edges(num_keypoints: int) -> Sequence[Tuple[int, int]]:
    """Return a lightweight skeleton topology for message passing."""
    if num_keypoints == 17:
        # COCO-style skeleton with additional symmetric links.
        return (
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7),
            (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16), (1, 2), (3, 4), (7, 8), (9, 10),
            (11, 12), (13, 14), (15, 16))
    # Fallback chain topology for arbitrary keypoint sets.
    return tuple((i, i + 1) for i in range(max(num_keypoints - 1, 0)))


class StructuralCritic(BaseModule):
    """Estimate per-joint error probability and displacement hint."""

    def __init__(self,
                 num_keypoints: int,
                 hidden_dim: int = 64,
                 edges: Optional[Sequence[Tuple[int, int]]] = None):
        super().__init__()
        self.num_keypoints = num_keypoints

        adj = torch.eye(num_keypoints, dtype=torch.float32)
        for i, j in (edges or _default_skeleton_edges(num_keypoints)):
            if 0 <= i < num_keypoints and 0 <= j < num_keypoints:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer('norm_adj', adj / deg)

        self.in_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.err_head = nn.Linear(hidden_dim, 1)
        self.disp_head = nn.Linear(hidden_dim, 2)

    def forward(self, pose_preds: Tensor, joint_scores: Tensor) -> Tuple[Tensor, Tensor]:
        # pose_preds: (..., K, 2), joint_scores: (..., K)
        critic_in = torch.cat((pose_preds, joint_scores.unsqueeze(-1)), dim=-1)
        hidden = self.in_proj(critic_in)
        hidden = torch.matmul(self.norm_adj.to(hidden), hidden)
        error_prob = torch.sigmoid(self.err_head(hidden)).squeeze(-1)
        disp_hint = self.disp_head(hidden)
        return error_prob, disp_hint


class FeatureRefinementActor(BaseModule):
    """Update keypoint latent features using critic feedback."""

    def __init__(self,
                 feat_dim: int,
                 hidden_dim: int = 128,
                 residual_scale: float = 0.5):
        super().__init__()
        self.err_proj = nn.Linear(1, feat_dim)
        self.disp_proj = nn.Linear(2, feat_dim)
        self.update = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.step_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, kpt_feats: Tensor, error_prob: Tensor,
                disp_hint: Tensor) -> Tensor:
        err_emb = self.err_proj(error_prob.unsqueeze(-1))
        disp_emb = self.disp_proj(disp_hint)
        delta = self.update(torch.cat((kpt_feats, err_emb, disp_emb), dim=-1))
        return kpt_feats + self.step_scale * torch.tanh(delta)


class ThinkCritiqueActLoop(BaseModule):
    """One step Think-Critique-Act refinement."""

    def __init__(self,
                 num_keypoints: int,
                 feat_dim: int,
                 critic_hidden_dim: int = 64,
                 actor_hidden_dim: int = 128,
                 residual_scale: float = 0.5):
        super().__init__()
        self.structural_critic = StructuralCritic(
            num_keypoints=num_keypoints, hidden_dim=critic_hidden_dim)
        self.feature_refinement_actor = FeatureRefinementActor(
            feat_dim=feat_dim,
            hidden_dim=actor_hidden_dim,
            residual_scale=residual_scale)

    def forward(self, kpt_feats: Tensor, pose_preds: Tensor,
                joint_scores: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        error_prob, disp_hint = self.structural_critic(pose_preds, joint_scores)
        refined_feats = self.feature_refinement_actor(kpt_feats, error_prob,
                                                      disp_hint)
        return refined_feats, error_prob, disp_hint
