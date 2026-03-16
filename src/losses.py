"""
losses.py — Custom loss functions for toxicity severity regression.

Three losses are implemented and can be combined:

  1. WeightedMSELoss      — per-category weighted mean squared error
  2. PairwiseRankingLoss  — encourages correct ordinal ranking between examples
  3. CombinedSeverityLoss — weighted sum of MSE + ranking (recommended)

The ranking loss is the key technical contribution: it trains the model to
not just predict the right absolute score, but to understand that comment A
is *more toxic* than comment B. This is especially useful when annotator
agreement is noisy (soft labels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Weighted MSE Loss
# ---------------------------------------------------------------------------

class WeightedMSELoss(nn.Module):
    """
    MSE loss with per-category weights.

    Rationale: threat and severe_toxicity are rare but high-stakes.
    Upweighting them prevents the model from ignoring them.

    Default weights correspond to:
        [toxicity, severe_toxicity, obscene, threat, insult, identity_attack]
    """

    DEFAULT_WEIGHTS = torch.tensor([1.0, 2.5, 1.0, 3.0, 1.0, 1.8])

    def __init__(self, weights: torch.Tensor = None):
        super().__init__()
        self.register_buffer(
            "weights",
            weights if weights is not None else self.DEFAULT_WEIGHTS,
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds   : (B, num_labels) — model outputs in [0, 1]
            targets : (B, num_labels) — soft label targets in [0, 1]
        Returns:
            scalar loss
        """
        squared_errors = (preds - targets) ** 2          # (B, num_labels)
        weighted = squared_errors * self.weights          # (B, num_labels)
        return weighted.mean()


# ---------------------------------------------------------------------------
# 2. Pairwise Ranking Loss
# ---------------------------------------------------------------------------

class PairwiseRankingLoss(nn.Module):
    """
    Encourages the model to preserve the ordinal ranking of toxicity scores.

    For every pair (i, j) in the batch:
      - If target[i] > target[j] + margin, then pred[i] should > pred[j]
      - Violations are penalized proportionally to how wrong the ranking is

    This is a margin-based loss similar to RankNet.

    Args:
        margin    : minimum score gap to enforce between pairs
        col_idx   : which label column to rank on (default 0 = overall toxicity)
        max_pairs : maximum pairs to sample per batch (for efficiency)
    """

    def __init__(self, margin: float = 0.1, col_idx: int = 0, max_pairs: int = 512):
        super().__init__()
        self.margin = margin
        self.col_idx = col_idx
        self.max_pairs = max_pairs

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds   : (B, num_labels)
            targets : (B, num_labels)
        Returns:
            scalar ranking loss
        """
        pred_scores = preds[:, self.col_idx]      # (B,)
        true_scores = targets[:, self.col_idx]    # (B,)

        B = pred_scores.size(0)

        # All pairwise differences: shape (B, B)
        pred_diff = pred_scores.unsqueeze(0) - pred_scores.unsqueeze(1)   # pred[i] - pred[j]
        true_diff = true_scores.unsqueeze(0) - true_scores.unsqueeze(1)   # true[i] - true[j]

        # Only penalize pairs where true ranking is clear (diff > margin)
        significant_pairs = true_diff.abs() > self.margin

        # Ranking violation: true says i > j but pred says i <= j (or vice versa)
        sign_agreement = pred_diff * true_diff.sign()
        violation = F.relu(self.margin - sign_agreement)

        loss = (violation * significant_pairs.float()).sum()
        n_pairs = significant_pairs.float().sum().clamp(min=1)
        return loss / n_pairs


# ---------------------------------------------------------------------------
# 3. Label Smoothing Regression Loss
# ---------------------------------------------------------------------------

class SmoothedL1Loss(nn.Module):
    """
    Huber loss (smooth L1) — less sensitive to outlier annotations than MSE.
    Useful because crowd-sourced labels can be noisy.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(preds, targets, beta=self.beta)


# ---------------------------------------------------------------------------
# 4. Combined Loss (recommended)
# ---------------------------------------------------------------------------

class CombinedSeverityLoss(nn.Module):
    """
    Combines weighted MSE + pairwise ranking loss.

    Loss = alpha * WeightedMSE + beta * RankingLoss

    This formulation:
      - Trains accurate absolute scores (MSE term)
      - Trains correct relative ordering (ranking term)
      - Upweights rare but serious categories (category weights)

    Args:
        alpha       : weight for MSE loss
        beta        : weight for ranking loss
        mse_weights : per-category weights for MSE
        rank_margin : margin for pairwise ranking
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        mse_weights: torch.Tensor = None,
        rank_margin: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = WeightedMSELoss(weights=mse_weights)
        self.rank_loss = PairwiseRankingLoss(margin=rank_margin)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Returns:
            total_loss  : combined scalar loss (used for backprop)
            loss_dict   : dict of individual components (for logging)
        """
        mse = self.mse_loss(preds, targets)
        rank = self.rank_loss(preds, targets)
        total = self.alpha * mse + self.beta * rank

        return total, {
            "loss": total.item(),
            "mse_loss": mse.item(),
            "rank_loss": rank.item(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss(config: dict) -> nn.Module:
    """Build loss function from config."""
    loss_type = config.get("loss", "combined")

    if loss_type == "mse":
        return WeightedMSELoss()
    elif loss_type == "ranking":
        return PairwiseRankingLoss()
    elif loss_type == "huber":
        return SmoothedL1Loss()
    elif loss_type == "combined":
        return CombinedSeverityLoss(
            alpha=config.get("loss_alpha", 1.0),
            beta=config.get("loss_beta", 0.3),
            rank_margin=config.get("rank_margin", 0.1),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
