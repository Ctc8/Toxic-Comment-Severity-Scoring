"""
train.py — Training loop for toxicity severity model.

Features:
  - Differential learning rates (lower LR for pretrained encoder)
  - Linear warmup + cosine decay scheduler
  - Mixed precision training (torch.cuda.amp)
  - Gradient clipping
  - Early stopping
  - Checkpoint saving (best val Pearson)
  - Weights & Biases logging (optional)
  - Ablation mode: specify config overrides via CLI

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --architecture baseline
    python src/train.py --config configs/config.yaml --loss mse --no-augment
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

from dataset import build_dataloaders
from evaluate import evaluate_model, print_summary
from losses import build_loss, CombinedSeverityLoss
from model import build_model, count_parameters


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Directories
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.train_loader, self.val_loader, self.tokenizer = build_dataloaders(config)

        # Model
        self.model = build_model(config).to(self.device)
        n_params = count_parameters(self.model)
        print(f"Trainable parameters: {n_params:,}")

        # Loss
        self.criterion = build_loss(config)

        # Optimizer — differential LRs
        encoder_params = list(self.model.encoder.parameters())
        head_params = [p for n, p in self.model.named_parameters()
                       if "encoder" not in n and p.requires_grad]

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": config.get("encoder_lr", 2e-5)},
            {"params": head_params,   "lr": config.get("head_lr", 1e-4)},
        ], weight_decay=config.get("weight_decay", 0.01))

        # Scheduler
        n_epochs = config.get("epochs", 3)
        n_steps = len(self.train_loader) * n_epochs
        warmup_steps = int(n_steps * config.get("warmup_ratio", 0.06))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=n_steps,
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=config.get("fp16", True) and torch.cuda.is_available())

        # State
        self.best_val_pearson = -1.0
        self.patience_counter = 0
        self.history = []

        # Optional W&B
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=config.get("wandb_project", "toxic-severity"), config=config)
                self.wandb = wandb
            except ImportError:
                print("wandb not installed — skipping logging.")
                self.use_wandb = False

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_rank = 0.0
        n_batches = len(self.train_loader)
        log_every = max(1, n_batches // 10)

        for step, batch in enumerate(self.train_loader):
            input_ids     = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels        = batch["labels"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                preds = self.model(input_ids=input_ids, attention_mask=attention_mask)

                if isinstance(self.criterion, CombinedSeverityLoss):
                    loss, loss_dict = self.criterion(preds, labels)
                    total_mse += loss_dict["mse_loss"]
                    total_rank += loss_dict["rank_loss"]
                else:
                    loss = self.criterion(preds, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

            if (step + 1) % log_every == 0:
                avg = total_loss / (step + 1)
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch} | Step {step+1}/{n_batches} | loss={avg:.4f} | lr={lr:.2e}")

        return {
            "train_loss": total_loss / n_batches,
            "train_mse":  total_mse / n_batches,
            "train_rank": total_rank / n_batches,
        }

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        all_preds, all_targets = [], []

        for batch in self.val_loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"]

            preds = self.model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())

        preds_np   = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)

        pearson  = pearsonr(preds_np[:, 0], targets_np[:, 0])[0]
        mae      = float(np.abs(preds_np[:, 0] - targets_np[:, 0]).mean())
        rmse     = float(np.sqrt(np.mean((preds_np[:, 0] - targets_np[:, 0]) ** 2)))

        return {"val_pearson": pearson, "val_mae": mae, "val_rmse": rmse}

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, metrics: dict):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        path = self.output_dir / "best_model.pt"
        torch.save(ckpt, path)
        print(f"  ✓ Saved best checkpoint to {path}")

    def load_best_checkpoint(self):
        path = self.output_dir / "best_model.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint (epoch {ckpt['epoch']})")

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def fit(self):
        n_epochs = self.config.get("epochs", 3)
        patience = self.config.get("early_stopping_patience", 2)

        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Architecture : {self.config.get('architecture', 'multitask')}")
        print(f"Loss         : {self.config.get('loss', 'combined')}")
        print(f"Model        : {self.config.get('model_name', 'roberta-base')}\n")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics   = self.validate()
            elapsed = time.time() - t0

            epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.history.append(epoch_metrics)

            val_pearson = val_metrics["val_pearson"]
            print(
                f"\nEpoch {epoch}/{n_epochs} ({elapsed:.0f}s) | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_pearson={val_pearson:.4f} | "
                f"val_mae={val_metrics['val_mae']:.4f}"
            )

            if self.use_wandb:
                self.wandb.log(epoch_metrics)

            if val_pearson > self.best_val_pearson:
                self.best_val_pearson = val_pearson
                self.patience_counter = 0
                self.save_checkpoint(epoch, epoch_metrics)
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{patience})")
                if self.patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nBest val Pearson: {self.best_val_pearson:.4f}")
        return self.history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train toxicity severity model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")

    # Allow override of any config value from CLI (for ablations)
    parser.add_argument("--architecture", choices=["baseline", "attention", "multitask"])
    parser.add_argument("--loss", choices=["mse", "ranking", "huber", "combined"])
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--freeze-layers", type=int)
    parser.add_argument("--sample-frac", type=float, help="Fraction of data to use (for fast experiments)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.architecture:  config["architecture"] = args.architecture
    if args.loss:          config["loss"] = args.loss
    if args.model_name:    config["model_name"] = args.model_name
    if args.epochs:        config["epochs"] = args.epochs
    if args.batch_size:    config["batch_size"] = args.batch_size
    if args.no_augment:    config["augment"] = False
    if args.freeze_layers: config["freeze_layers"] = args.freeze_layers
    if args.sample_frac:   config["sample_frac"] = args.sample_frac

    trainer = Trainer(config)
    trainer.fit()

    # Final full evaluation with best checkpoint
    print("\nRunning final evaluation with best checkpoint...")
    trainer.load_best_checkpoint()
    results = evaluate_model(trainer.model, trainer.val_loader, trainer.device)
    print_summary(results)


if __name__ == "__main__":
    main()
