"""
model.py — Toxicity Severity Model Architectures

Three tiers:
  - BaselineToxicityModel   : CLS-pooled RoBERTa + linear regressor
  - AttentionToxicityModel  : Token-level attention pooling + explainability
  - MultiTaskToxicityModel  : Joint overall severity + per-category heads (recommended)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


# ---------------------------------------------------------------------------
# Tier 1 — Baseline
# ---------------------------------------------------------------------------

class BaselineToxicityModel(nn.Module):
    """
    Fine-tunes a pretrained encoder (BERT/RoBERTa/DeBERTa) with a simple
    linear regression head on top of the [CLS] token.

    Predicts 6 severity scores in [0, 1]:
        [toxicity, severe_toxicity, obscene, threat, insult, identity_attack]
    """

    def __init__(self, model_name: str = "roberta-base", num_labels: int = 6, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.regressor(self.dropout(cls_output))
        return torch.sigmoid(logits)  # shape: (B, num_labels)


# ---------------------------------------------------------------------------
# Tier 2 — Attention Pooling Head (+ explainability)
# ---------------------------------------------------------------------------

class TokenAttentionPooling(nn.Module):
    """
    Learns a scalar attention weight for each token, then computes
    a weighted sum of token embeddings. Returns both the pooled vector
    and the attention weights (useful for visualization).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn_fc = nn.Linear(hidden_size, 1)

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            token_embeddings : (B, L, H)
            attention_mask   : (B, L)  — 1 for real tokens, 0 for padding
        Returns:
            pooled  : (B, H)
            weights : (B, L)  — normalized attention over real tokens
        """
        scores = self.attn_fc(token_embeddings).squeeze(-1)            # (B, L)
        scores = scores.masked_fill(attention_mask == 0, float("-inf")) # mask padding
        weights = torch.softmax(scores, dim=-1)                         # (B, L)
        pooled = (weights.unsqueeze(-1) * token_embeddings).sum(dim=1) # (B, H)
        return pooled, weights


class AttentionToxicityModel(nn.Module):
    """
    Replaces CLS pooling with learned token-level attention.
    Enables per-token attribution: which words drove the severity score?
    """

    def __init__(self, model_name: str = "roberta-base", num_labels: int = 6, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.attention_pool = TokenAttentionPooling(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_weights=False):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled, attn_weights = self.attention_pool(outputs.last_hidden_state, attention_mask)
        logits = self.regressor(self.dropout(pooled))
        preds = torch.sigmoid(logits)

        if return_weights:
            return preds, attn_weights  # attn_weights used for visualization
        return preds


# ---------------------------------------------------------------------------
# Tier 3 — Multi-Task Model (recommended for best results)
# ---------------------------------------------------------------------------

class MultiTaskToxicityModel(nn.Module):
    """
    Joint prediction of:
      - overall_severity  : single scalar (primary task)
      - category_scores   : 5 sub-category scores (auxiliary task)

    Both heads share the same transformer encoder backbone.
    Multi-task training acts as regularization and improves generalization.

    Architecture:
        Encoder → TokenAttentionPooling
                → [overall_head]    → overall severity (1 value)
                → [category_head]   → per-category scores (5 values)
    """

    CATEGORIES = ["severe_toxicity", "obscene", "threat", "insult", "identity_attack"]

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.3,
        freeze_layers: int = 0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Optionally freeze bottom N transformer layers for efficient fine-tuning
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        self.attention_pool = TokenAttentionPooling(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Shared projection
        self.shared_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # Task-specific heads
        self.overall_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.CATEGORIES)),
        )

    def _freeze_layers(self, num_layers: int):
        """Freeze the embedding layer and first `num_layers` transformer blocks."""
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        for layer in self.encoder.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_weights=False):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled, attn_weights = self.attention_pool(outputs.last_hidden_state, attention_mask)
        shared = self.shared_proj(self.dropout(pooled))

        overall = torch.sigmoid(self.overall_head(shared))          # (B, 1)
        categories = torch.sigmoid(self.category_head(shared))      # (B, 5)
        preds = torch.cat([overall, categories], dim=-1)            # (B, 6)

        if return_weights:
            return preds, attn_weights
        return preds


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: dict) -> nn.Module:
    """Factory function — reads from config dict."""
    arch = config.get("architecture", "multitask")
    model_name = config.get("model_name", "roberta-base")
    dropout = config.get("dropout", 0.3)

    if arch == "baseline":
        return BaselineToxicityModel(model_name=model_name, dropout=dropout)
    elif arch == "attention":
        return AttentionToxicityModel(model_name=model_name, dropout=dropout)
    elif arch == "multitask":
        return MultiTaskToxicityModel(
            model_name=model_name,
            dropout=dropout,
            freeze_layers=config.get("freeze_layers", 0),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
