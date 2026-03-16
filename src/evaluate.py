"""
evaluate.py — Evaluation metrics for toxicity severity prediction.

Metrics implemented:
  - Pearson / Spearman correlation (standard for Jigsaw)
  - MAE, RMSE per category
  - AUC-ROC (binarized at 0.5 for comparison with classifiers)
  - Bias audit: per-identity-group score gap analysis
  - Calibration: reliability diagram data
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, mean_absolute_error
from typing import Dict, Optional
import torch


LABEL_COLS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]

IDENTITY_COLS = [
    "male", "female", "transgender",
    "christian", "jewish", "muslim",
    "black", "white", "asian", "latino",
    "psychiatric_or_mental_illness",
]


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def pearson_correlation(preds: np.ndarray, targets: np.ndarray, col: int = 0) -> float:
    """Pearson r between predicted and true severity scores."""
    return stats.pearsonr(preds[:, col], targets[:, col])[0]


def spearman_correlation(preds: np.ndarray, targets: np.ndarray, col: int = 0) -> float:
    """Spearman ρ — rank-based, robust to outliers."""
    return stats.spearmanr(preds[:, col], targets[:, col])[0]


def per_category_metrics(preds: np.ndarray, targets: np.ndarray, label_cols=LABEL_COLS) -> pd.DataFrame:
    """
    Compute MAE, RMSE, Pearson, Spearman for each label column.

    Returns a DataFrame with one row per category.
    """
    rows = []
    for i, col in enumerate(label_cols):
        p, t = preds[:, i], targets[:, i]
        rows.append({
            "category": col,
            "mae": mean_absolute_error(t, p),
            "rmse": np.sqrt(np.mean((p - t) ** 2)),
            "pearson": stats.pearsonr(p, t)[0],
            "spearman": stats.spearmanr(p, t)[0],
        })
    return pd.DataFrame(rows).set_index("category")


def auc_roc(preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Binarize targets at threshold, then compute AUC-ROC.
    Allows comparison with binary classifiers in related work.
    """
    results = {}
    for i, col in enumerate(LABEL_COLS):
        binary_targets = (targets[:, i] >= threshold).astype(int)
        if binary_targets.sum() > 0 and binary_targets.sum() < len(binary_targets):
            results[col] = roc_auc_score(binary_targets, preds[:, i])
        else:
            results[col] = float("nan")
    return results


# ---------------------------------------------------------------------------
# Bias Audit
# ---------------------------------------------------------------------------

def bias_audit(
    preds: np.ndarray,
    targets: np.ndarray,
    df: pd.DataFrame,
    identity_cols: list = IDENTITY_COLS,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Measure whether the model assigns systematically different scores to
    comments mentioning specific identity groups.

    For each identity group we compute:
      - subgroup_auc     : AUC on comments mentioning this group
      - bpsn_auc         : Background Positive Subgroup Negative AUC
                           (non-toxic background + toxic subgroup)
      - bnsp_auc         : Background Negative Subgroup Positive AUC
                           (toxic background + non-toxic subgroup)

    These three metrics are used in the original Jigsaw competition scoring.
    A model with disparate subgroup AUCs is exhibiting identity bias.

    Args:
        preds    : (N, 6) model predictions
        targets  : (N, 6) ground truth
        df       : original DataFrame with identity columns
    """
    overall_toxic = targets[:, 0]
    pred_overall = preds[:, 0]
    rows = []

    for col in identity_cols:
        if col not in df.columns:
            continue

        subgroup_mask = (df[col].fillna(0) >= 0.5).values
        if subgroup_mask.sum() < 10:
            continue  # skip groups with very few examples

        # --- Subgroup AUC: model performance within this group ---
        sg_true = (overall_toxic[subgroup_mask] >= 0.5).astype(int)
        sg_pred = pred_overall[subgroup_mask]
        if sg_true.sum() > 0 and sg_true.sum() < subgroup_mask.sum():
            sg_auc = roc_auc_score(sg_true, sg_pred)
        else:
            sg_auc = float("nan")

        # --- BPSN: background (non-subgroup) positives + subgroup negatives ---
        bpsn_mask = (~subgroup_mask & (overall_toxic >= 0.5)) | \
                    (subgroup_mask & (overall_toxic < 0.5))
        bpsn_true = (overall_toxic[bpsn_mask] >= 0.5).astype(int)
        bpsn_pred = pred_overall[bpsn_mask]
        if bpsn_true.sum() > 0 and bpsn_true.sum() < bpsn_mask.sum():
            bpsn_auc = roc_auc_score(bpsn_true, bpsn_pred)
        else:
            bpsn_auc = float("nan")

        # --- BNSP: background negatives + subgroup positives ---
        bnsp_mask = (~subgroup_mask & (overall_toxic < 0.5)) | \
                    (subgroup_mask & (overall_toxic >= 0.5))
        bnsp_true = (overall_toxic[bnsp_mask] >= 0.5).astype(int)
        bnsp_pred = pred_overall[bnsp_mask]
        if bnsp_true.sum() > 0 and bnsp_true.sum() < bnsp_mask.sum():
            bnsp_auc = roc_auc_score(bnsp_true, bnsp_pred)
        else:
            bnsp_auc = float("nan")

        rows.append({
            "identity": col,
            "n_examples": int(subgroup_mask.sum()),
            "subgroup_auc": sg_auc,
            "bpsn_auc": bpsn_auc,
            "bnsp_auc": bnsp_auc,
            "mean_pred_score": float(sg_pred.mean()),
        })

    return pd.DataFrame(rows).set_index("identity")


def jigsaw_final_score(bias_df: pd.DataFrame, overall_auc: float) -> float:
    """
    Official Jigsaw competition metric:
        Final = w * overall_AUC + (1-w) * mean(generalized_mean(subgroup_AUCs))
    where w = 0.25 and generalized mean power p = -5.
    """
    w = 0.25
    p = -5  # power for generalized mean (heavily penalizes worst groups)

    aucs = []
    for col in ["subgroup_auc", "bpsn_auc", "bnsp_auc"]:
        vals = bias_df[col].dropna().values
        if len(vals) > 0:
            gm = (np.mean(vals ** p)) ** (1 / p)
            aucs.append(gm)

    bias_score = np.mean(aucs) if aucs else 0.0
    return w * overall_auc + (1 - w) * bias_score


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_data(preds: np.ndarray, targets: np.ndarray, n_bins: int = 10, col: int = 0):
    """
    Compute data for a reliability diagram.
    Returns (bin_centers, mean_pred, mean_true, counts) for plotting.
    """
    p, t = preds[:, col], targets[:, col]
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, mean_pred, mean_true, counts = [], [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            mean_pred.append(p[mask].mean())
            mean_true.append(t[mask].mean())
            counts.append(mask.sum())

    return np.array(bin_centers), np.array(mean_pred), np.array(mean_true), np.array(counts)


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, dataloader, device, df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Run full evaluation on a dataloader.

    Returns a dict with all metrics and raw predictions/targets.
    """
    model.eval()
    all_preds, all_targets, all_idxs = [], [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.numpy())
        all_idxs.extend(batch["idx"].tolist())

    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    # --- Core metrics ---
    results = {
        "pearson": pearson_correlation(preds_np, targets_np),
        "spearman": spearman_correlation(preds_np, targets_np),
        "mae": mean_absolute_error(targets_np[:, 0], preds_np[:, 0]),
        "rmse": float(np.sqrt(np.mean((preds_np[:, 0] - targets_np[:, 0]) ** 2))),
    }

    results["per_category"] = per_category_metrics(preds_np, targets_np)
    results["auc_roc"] = auc_roc(preds_np, targets_np)

    # --- Bias audit (requires original df) ---
    if df is not None:
        subset_df = df.iloc[all_idxs].reset_index(drop=True)
        bias_df = bias_audit(preds_np, targets_np, subset_df)
        results["bias_audit"] = bias_df
        overall_auc = np.nanmean(list(results["auc_roc"].values()))
        results["jigsaw_score"] = jigsaw_final_score(bias_df, overall_auc)

    results["preds"] = preds_np
    results["targets"] = targets_np

    return results


def print_summary(results: Dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Pearson (overall):   {results['pearson']:.4f}")
    print(f"  Spearman (overall):  {results['spearman']:.4f}")
    print(f"  MAE (overall):       {results['mae']:.4f}")
    print(f"  RMSE (overall):      {results['rmse']:.4f}")
    if "jigsaw_score" in results:
        print(f"  Jigsaw Final Score:  {results['jigsaw_score']:.4f}")
    print("-" * 55)
    print("  Per-Category Metrics:")
    print(results["per_category"].to_string())
    if "bias_audit" in results:
        print("-" * 55)
        print("  Bias Audit (subgroup AUC):")
        print(results["bias_audit"][["subgroup_auc", "bpsn_auc", "bnsp_auc"]].to_string())
    print("=" * 55 + "\n")
