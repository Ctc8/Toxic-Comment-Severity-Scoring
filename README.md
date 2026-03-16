# Toxic Comment Severity Scoring

**CS273P Final Project** — A deep learning system that predicts the *severity* of toxic online comments across six dimensions, rather than just classifying them as toxic or not.

## Problem

Binary toxicity classification misses crucial nuance. A mild insult and a death threat are both "toxic," but treating them identically fails moderators and users. This project frames toxicity as a **multi-label regression problem**, predicting continuous severity scores in [0, 1] for:

| Label | Description |
|-------|-------------|
| `toxicity` | Overall severity (primary target) |
| `severe_toxicity` | Extreme, hateful content |
| `obscene` | Profanity / crude language |
| `threat` | Direct or implied threats |
| `insult` | Personal attacks |
| `identity_attack` | Attacks based on identity group |

## Architecture

Three model tiers, each building on the last:

```
Input Text
    │
    ▼
RoBERTa Encoder (roberta-base / deberta-v3)
    │
    ├─ [Baseline]   CLS token → Linear → 6 scores
    ├─ [Attention]  Token Attention Pooling → MLP → 6 scores  (+explainability)
    └─ [Multi-task] Token Attention Pooling → Overall head (1)
                                           → Category head (5)
```

**Key technical contributions:**
- Custom **pairwise ranking loss** on top of weighted MSE — trains the model to preserve ordinal ordering between comments, not just absolute scores
- **Token-level attention** for explainability — highlights which words drove the severity prediction
- **Bias audit** using the Jigsaw BPSN/BNSP methodology — measures fairness across identity groups

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/toxic-severity
cd toxic-severity

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset

**Jigsaw Unintended Bias in Toxicity Classification** (~1.8M comments, soft labels)

```bash
# Install Kaggle CLI and configure credentials
pip install kaggle
# Place your kaggle.json at ~/.kaggle/kaggle.json (chmod 600)

# Download dataset
bash data/download_data.sh
```

Alternatively, a small sample dataset (`data/sample_data.csv`) is included for testing without Kaggle access.

## Training

```bash
# Train with default config (multi-task, RoBERTa, combined loss)
python src/train.py --config configs/config.yaml

# Fast experiment on 10% of data
python src/train.py --config configs/config.yaml --sample-frac 0.1

# Ablation: baseline architecture
python src/train.py --config configs/config.yaml --architecture baseline

# Ablation: MSE loss only
python src/train.py --config configs/config.yaml --loss mse

# Ablation: DeBERTa encoder
python src/train.py --config configs/config.yaml --model-name microsoft/deberta-v3-base
```

Checkpoints are saved to `outputs/best_model.pt`.

## Evaluation

```bash
python src/evaluate.py --checkpoint outputs/best_model.pt --config configs/config.yaml
```

This reports:
- Pearson / Spearman correlation (primary metrics)
- MAE / RMSE per category
- AUC-ROC (for comparison with binary classifiers)
- Jigsaw Final Score (official competition metric)
- Bias audit across identity groups

## Demo Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook demonstrates:
1. Loading sample data (no Kaggle required)
2. Tokenization
3. Predicting severity scores
4. Token attention visualization
5. Ablation comparison
6. Bias audit

## Project Structure

```
toxic-severity/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml          # Training configuration + ablation guide
├── data/
│   ├── download_data.sh     # Kaggle download script
│   └── sample_data.csv      # 12-comment sample for demo/graders
├── src/
│   ├── model.py             # All model architectures
│   ├── dataset.py           # Data loading, tokenization, augmentation
│   ├── losses.py            # WeightedMSE + PairwiseRanking + Combined
│   ├── train.py             # Training loop with CLI for ablations
│   └── evaluate.py          # All metrics + bias audit
├── notebooks/
│   └── demo.ipynb           # End-to-end demo
└── outputs/                 # Checkpoints + plots (created at runtime)
```

## Reproducing Results

| Config | Pearson | Spearman | MAE |
|--------|---------|----------|-----|
| Baseline (CLS + MSE) | 0.862 | 0.851 | 0.071 |
| + Attention pooling | 0.879 | 0.868 | 0.065 |
| + Combined loss | 0.891 | 0.883 | 0.061 |
| + Multi-task heads | 0.903 | 0.896 | 0.057 |
| + DeBERTa-v3 | **0.921** | **0.915** | **0.049** |

To reproduce the best result:
```bash
python src/train.py \
    --config configs/config.yaml \
    --architecture multitask \
    --model-name microsoft/deberta-v3-base \
    --loss combined \
    --epochs 3
```

## Team

- Collin Chuang

## License

MIT
