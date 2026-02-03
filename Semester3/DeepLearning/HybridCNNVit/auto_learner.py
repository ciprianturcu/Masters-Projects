import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.abspath(os.path.join(_HERE, "..")))
from model import HybridCNNViT
from evolution import get_default_config, mutate_config
from dataset import NIHChestXrayDataset, LABELS

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

_PARENT = os.path.abspath(os.path.join(_HERE, ".."))

DATA_DIR = os.path.join(_PARENT, "datasets", "nih-chest-xrays", "data", "versions", "3")
DATA_DIR_MEMMAP = os.path.join(_PARENT, "datasets", "nih-chest-xrays", "data", "versions", "3_memmap")

SAVE_DIR = os.path.join(_HERE, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

STATE_FILE = os.path.join(SAVE_DIR, "auto_learner_state.pth")
CURRENT_EXP_FILE = os.path.join(SAVE_DIR, "auto_current_experiment.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "auto_best_model.pth")
LOG_FILE = os.path.join(_HERE, "auto_training_log.xlsx")


def parse_args():
    parser = argparse.ArgumentParser(description="Evolutionary HP search for Hybrid CNN-ViT")
    parser.add_argument("--dry-run", action="store_true", help="Quick test with 200 images")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per generation")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--memmap", action="store_true", help="Use memmap data")
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def get_dataloaders(config, args, split_datasets=None):
    if split_datasets is None:
        memmap_dir = DATA_DIR_MEMMAP if args.memmap else None
        data_dir = DATA_DIR_MEMMAP if args.memmap else DATA_DIR
        train_set = NIHChestXrayDataset(data_dir, split="train", image_size=args.image_size,
                                        memmap_dir=memmap_dir)
        val_set = NIHChestXrayDataset(data_dir, split="val", image_size=args.image_size,
                                      memmap_dir=memmap_dir)
    else:
        train_set, val_set = split_datasets

    if args.dry_run:
        print("  >> DRY RUN: Limiting to 200 images.")
        train_set = Subset(train_set, range(min(200, len(train_set))))
        val_set = Subset(val_set, range(min(200, len(val_set))))

    bs = config["batch_size"]
    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=persistent, prefetch_factor=4 if persistent else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=4 if persistent else None,
    )
    return train_loader, val_loader


def build_model(config, device, image_size=224):
    model = HybridCNNViT(
        num_labels=len(LABELS),
        cnn_channels=config.get("cnn_channels", [32, 64, 128, 256, 384]),
        embed_dim=config.get("embed_dim", 384),
        trans_depth=config.get("trans_depth", 6),
        num_heads=config.get("num_heads", 6),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        drop=config.get("dropout", 0.0),
        image_size=image_size,
    ).to(device)
    return model


def build_optimizer(model, config):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if config["optimizer"] == "adamw":
        return torch.optim.AdamW([
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config["lr"])
    else:
        return torch.optim.Adam(model.parameters(), lr=config["lr"],
                                weight_decay=config["weight_decay"])


def build_scheduler(optimizer, config, epochs):
    if config["scheduler"] == "cosine":
        warmup_epochs = min(3, epochs // 2)
        if warmup_epochs > 0 and epochs > warmup_epochs:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, cosine], milestones=[warmup_epochs])
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif config["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2)
    return None


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []
    for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    per_label_auc = {}
    for i, name in enumerate(LABELS):
        if len(np.unique(all_labels[:, i])) > 1:
            per_label_auc[name] = roc_auc_score(all_labels[:, i], all_probs[:, i])
        else:
            per_label_auc[name] = float("nan")
    valid = [v for v in per_label_auc.values() if not np.isnan(v)]
    mean_auc = np.mean(valid) if valid else 0.0
    return avg_loss, mean_auc, per_label_auc


def log_experiment(output_file, config, metrics):
    import pandas as pd
    from datetime import datetime

    config_flat = {}
    for k, v in config.items():
        if isinstance(v, list):
            config_flat[k] = str(v)
        else:
            config_flat[k] = v

    data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": "HybridCNNViT",
        **config_flat,
        **metrics,
    }
    df_new = pd.DataFrame([data])

    if os.path.exists(output_file):
        try:
            df_existing = pd.read_excel(output_file)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_final = df_new
    else:
        df_final = df_new

    df_final.to_excel(output_file, index=False)
    print(f"  Logged results to {output_file}")


def load_overall_state():
    if os.path.exists(STATE_FILE):
        return torch.load(STATE_FILE, weights_only=False)
    return {"best_auc": 0.0, "best_config": get_default_config(), "generation": 0}


def save_overall_state(best_auc, best_config, generation):
    torch.save({
        "best_auc": best_auc,
        "best_config": best_config,
        "generation": generation,
    }, STATE_FILE)


def save_experiment_state(model, optimizer, epoch, config, val_auc, generation):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "val_auc": val_auc,
        "generation": generation,
    }, CURRENT_EXP_FILE)


def run_training_cycle(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    global_state = load_overall_state()
    best_config = global_state["best_config"]
    best_auc_overall = global_state["best_auc"]
    generation = global_state["generation"]

    current_config = best_config
    start_epoch = 0
    architecture_changed = False

    resume_mode = False
    if os.path.exists(CURRENT_EXP_FILE) and not args.dry_run:
        print("\n>> FOUND INTERRUPTED EXPERIMENT. RESUMING...")
        checkpoint = torch.load(CURRENT_EXP_FILE, weights_only=False)
        current_config = checkpoint["config"]
        start_epoch = checkpoint["epoch"] + 1
        resume_mode = True
        generation = checkpoint.get("generation", generation)
    else:
        if generation > 0:
            print(f"\n>> Generation {generation + 1}: Mutating best config...")
            current_config = mutate_config(best_config)
            architecture_changed = current_config.pop("architecture_changed", False)
        else:
            print("\n>> Generation 1: Baseline config")

    print(f"\nConfig: {json.dumps(current_config, indent=2, default=str)}")
    print(f"Best AUC so far: {best_auc_overall:.4f}")
    if architecture_changed:
        print("  (Architecture changed — fresh initialization)")

    model = build_model(current_config, device, args.image_size)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {trainable_params:,} trainable / {total_params:,} total")

    optimizer = build_optimizer(model, current_config)
    scheduler = build_scheduler(optimizer, current_config, args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda")

    if resume_mode:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    elif not architecture_changed and generation > 0 and os.path.exists(BEST_MODEL_PATH):
        try:
            best_state = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
            model.load_state_dict(best_state)
            print("  Loaded best model weights (training HP mutation only)")
        except Exception as e:
            print(f"  Could not load best model weights ({e}), starting fresh")

    train_loader, val_loader = get_dataloaders(current_config, args)

    best_auc_this_run = 0.0
    target_epochs = 1 if args.dry_run else args.epochs

    print(f"\nTraining for {target_epochs} epochs (gen {generation + 1})...\n")

    try:
        for epoch in range(start_epoch, target_epochs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device)

            val_loss, val_auc, val_per_label = evaluate(
                model, val_loader, criterion, device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            if scheduler is not None:
                if current_config["scheduler"] == "plateau":
                    scheduler.step(val_auc)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Gen {generation + 1} | Epoch {epoch + 1}/{target_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.0f}s")

            if val_auc > best_auc_this_run:
                best_auc_this_run = val_auc

            if not args.dry_run:
                save_experiment_state(model, optimizer, epoch, current_config, val_auc, generation)

    except KeyboardInterrupt:
        print("\nInterrupted! State saved. Run again to resume.")
        return False

    print(f"\n>> Generation {generation + 1} finished. Best AUC: {best_auc_this_run:.4f}")

    improved = False
    if best_auc_this_run > best_auc_overall:
        print(f"  !!! NEW RECORD !!! {best_auc_overall:.4f} -> {best_auc_this_run:.4f}")
        best_auc_overall = best_auc_this_run
        best_config = current_config
        if not args.dry_run:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        improved = True
    else:
        print(f"  No improvement over {best_auc_overall:.4f}. Reverting config.")

    metrics = {
        "Val AUC": best_auc_this_run,
        "Improved": improved,
        "Generation": generation + 1,
    }
    log_experiment(LOG_FILE, current_config, metrics)

    if os.path.exists(CURRENT_EXP_FILE) and not args.dry_run:
        os.remove(CURRENT_EXP_FILE)

    if not args.dry_run:
        save_overall_state(best_auc_overall, best_config, generation + 1)

    if args.dry_run:
        print("\nDry run complete.")
        return False

    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("  Hybrid CNN-ViT Evolutionary Hyperparameter Search")
    print("=" * 60)

    while True:
        should_continue = run_training_cycle(args)
        if not should_continue:
            break
        print("\n>> Restarting cycle...\n")


if __name__ == "__main__":
    main()
