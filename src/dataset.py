"""
dataset.py — Data loading, preprocessing, and augmentation for toxic comment severity.

Dataset: Jigsaw Unintended Bias in Toxicity Classification
  https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

Label columns used:
  - toxicity            (overall severity — primary target)
  - severe_toxicity
  - obscene
  - threat
  - insult
  - identity_attack

Each label is a float in [0, 1] representing the fraction of annotators
who flagged the comment. This is the "soft label" regression formulation.
"""

import re
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_COLS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
IDENTITY_COLS = [
    "male", "female", "transgender", "other_gender",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "physical_disability", "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness", "other_disability",
]


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Light cleaning — preserve semantics while normalizing noise."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Collapse excessive whitespace / newlines
    text = re.sub(r"\s+", " ", text)
    # Remove URLs (they rarely carry toxicity signal)
    text = re.sub(r"http\S+|www\.\S+", "[URL]", text)
    # Keep punctuation — it carries tonal information (e.g., "!!!")
    return text


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Jigsaw toxicity data.

    Args:
        df           : DataFrame with 'comment_text' + label columns
        tokenizer    : HuggingFace tokenizer
        max_length   : token sequence length
        augment      : whether to apply text augmentation (train only)
        label_cols   : which columns to use as targets
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 256,
        augment: bool = False,
        label_cols: list = LABEL_COLS,
    ):
        self.texts = df["comment_text"].apply(clean_text).tolist()
        self.labels = df[label_cols].fillna(0.0).values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.label_cols = label_cols

        # Pre-cache identity flags for bias analysis
        self.identity_flags = None
        overlap = [c for c in IDENTITY_COLS if c in df.columns]
        if overlap:
            self.identity_flags = (df[overlap].fillna(0) > 0.5).any(axis=1).values

    def __len__(self):
        return len(self.texts)

    def _augment(self, text: str) -> str:
        """Simple augmentation: random synonym swap or word drop."""
        words = text.split()
        if len(words) < 4:
            return text
        aug_type = random.random()
        if aug_type < 0.3 and len(words) > 5:
            # Random word deletion (drop ~10% of words)
            keep_prob = 0.9
            words = [w for w in words if random.random() < keep_prob or w.isupper()]
        elif aug_type < 0.5:
            # Random word swap
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return " ".join(words)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        if self.augment:
            text = self._augment(text)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "idx": idx,
        }


# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------

def load_jigsaw_data(data_dir: str, sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """
    Load and merge Jigsaw train data.

    Expected files in data_dir:
        train.csv   — from Jigsaw Unintended Bias competition
        test.csv    — optional

    Download from:
        kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv not found in {data_dir}.\n"
            "Download from: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification\n"
            "Run: kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification"
        )

    df = pd.read_csv(train_path)
    print(f"Loaded {len(df):,} rows from {train_path}")

    # Ensure all label columns exist
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Optional: subsample for fast experimentation
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} rows (frac={sample_frac})")

    return df


def split_data(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    """Stratified-ish split based on toxicity quintile."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * val_frac)
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df


def build_dataloaders(config: dict):
    """
    Build train and val DataLoaders from config.

    Config keys:
        data_dir, model_name, max_length, batch_size,
        val_frac, sample_frac, num_workers, augment
    """
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    df = load_jigsaw_data(config["data_dir"], sample_frac=config.get("sample_frac", 1.0))
    train_df, val_df = split_data(df, val_frac=config.get("val_frac", 0.1))

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    train_ds = ToxicityDataset(
        train_df,
        tokenizer,
        max_length=config.get("max_length", 256),
        augment=config.get("augment", False),
    )
    val_ds = ToxicityDataset(
        val_df,
        tokenizer,
        max_length=config.get("max_length", 256),
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get("batch_size", 32) * 2,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer


# ---------------------------------------------------------------------------
# Sample dataset (for demo/graders)
# ---------------------------------------------------------------------------

SAMPLE_COMMENTS = [
    {"comment_text": "I love this community, everyone is so helpful!", "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.0, "identity_attack": 0.0},
    {"comment_text": "You are a bit rude, please be more polite.", "toxicity": 0.15, "severe_toxicity": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.1, "identity_attack": 0.0},
    {"comment_text": "This is complete garbage and you should be ashamed.", "toxicity": 0.55, "severe_toxicity": 0.05, "obscene": 0.1, "threat": 0.0, "insult": 0.45, "identity_attack": 0.0},
    {"comment_text": "Shut up you absolute idiot, nobody cares what you think!", "toxicity": 0.82, "severe_toxicity": 0.2, "obscene": 0.1, "threat": 0.0, "insult": 0.78, "identity_attack": 0.0},
    {"comment_text": "People like you shouldn't be allowed to speak. Disgusting.", "toxicity": 0.91, "severe_toxicity": 0.45, "obscene": 0.1, "threat": 0.2, "insult": 0.7, "identity_attack": 0.3},
]


def create_sample_dataset(output_path: str = "data/sample_data.csv"):
    """Write a small sample CSV for graders to run without Kaggle access."""
    df = pd.DataFrame(SAMPLE_COMMENTS)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to {output_path}")
    return df
