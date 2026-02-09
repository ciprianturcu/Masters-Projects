import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.abspath(os.path.join(_HERE, "..")))
from model import HybridCNNViT
from dataset import NIHChestXrayDataset, LABELS

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

_PARENT = os.path.abspath(os.path.join(_HERE, ".."))

DATA_DIR_ORIGINAL = os.path.join(_PARENT, "datasets", "nih-chest-xrays", "data", "versions", "3")
DATA_DIR_RESIZED = os.path.join(_PARENT, "datasets", "nih-chest-xrays", "data", "versions", "3_resized")
DATA_DIR_JPG = os.path.join(_PARENT, "datasets", "nih-chest-xrays", "data", "versions", "3_jpg")
DATA_DIR_MEMMAP = os.path.join(_PARENT, "datasets", "nih-chest-xrays", "data", "versions", "3_memmap")

SAVE_DIR = os.path.join(_HERE, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid CNN-ViT on NIH-CXR14")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (higher than LT-ViT since training from scratch)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--trans_depth", type=int, default=6,
                        help="Number of transformer encoder blocks")
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (longer than LT-ViT since from-scratch)")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Linear warmup epochs before cosine annealing")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Use pre-resized images from 3_resized/")
    parser.add_argument("--jpg", action="store_true",
                        help="Use JPEG images from 3_jpg/")
    parser.add_argument("--memmap", action="store_true",
                        help="Use memmap images from 3_memmap/")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for faster training")
    return parser.parse_args()


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
    all_labels = []
    all_probs = []

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
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    per_label_auc = {}
    for i, label_name in enumerate(LABELS):
        if len(np.unique(all_labels[:, i])) > 1:
            per_label_auc[label_name] = roc_auc_score(all_labels[:, i], all_probs[:, i])
        else:
            per_label_auc[label_name] = float("nan")

    valid_aucs = [v for v in per_label_auc.values() if not np.isnan(v)]
    mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0

    return avg_loss, mean_auc, per_label_auc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    memmap_dir = DATA_DIR_MEMMAP if args.memmap else None
    if args.memmap:
        data_dir = DATA_DIR_MEMMAP
    elif args.jpg:
        data_dir = DATA_DIR_JPG
    elif args.preprocessed:
        data_dir = DATA_DIR_RESIZED
    else:
        data_dir = DATA_DIR_ORIGINAL
    print(f"\nLoading datasets from: {data_dir}")

    train_set = NIHChestXrayDataset(data_dir, split="train", image_size=args.image_size,
                                    preprocessed=args.preprocessed, memmap_dir=memmap_dir)
    val_set = NIHChestXrayDataset(data_dir, split="val", image_size=args.image_size,
                                  preprocessed=args.preprocessed, memmap_dir=memmap_dir)
    test_set = NIHChestXrayDataset(data_dir, split="test", image_size=args.image_size,
                                   preprocessed=args.preprocessed, memmap_dir=memmap_dir)

    print(f"  Train: {len(train_set)} images")
    print(f"  Val:   {len(val_set)} images")
    print(f"  Test:  {len(test_set)} images")

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=persistent, prefetch_factor=4 if persistent else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=4 if persistent else None,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=4 if persistent else None,
    )

    print("\nBuilding Hybrid CNN-ViT model (from scratch)...")
    model = HybridCNNViT(
        num_labels=len(LABELS),
        embed_dim=args.embed_dim,
        trans_depth=args.trans_depth,
        num_heads=args.num_heads,
        drop=args.dropout,
        image_size=args.image_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    if args.compile:
        try:
            model = torch.compile(model, backend="cudagraphs")
            print("  Model compiled with cudagraphs backend")
        except Exception as e:
            print(f"  torch.compile() failed ({e}), continuing without compilation")

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda")

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    start_epoch = 0
    best_auc = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best AUC so far: {best_auc:.4f}")

    run_tag = (f"hybrid_cnn_vit_bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}"
               f"_d{args.trans_depth}_h{args.num_heads}_img{args.image_size}")

    print(f"\nStarting training for {args.epochs} epochs...\n")
    print(f"Run tag: {run_tag}\n")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device)

        val_loss, val_auc, val_per_label = evaluate(
            model, val_loader, criterion, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - epoch_start

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1}/{args.epochs}  |  "
              f"Train Loss: {train_loss:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  |  "
              f"Val AUC: {val_auc:.4f}  |  "
              f"LR: {current_lr:.2e}  |  "
              f"Time: {elapsed:.0f}s")

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
            "val_auc": val_auc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": {
                "embed_dim": args.embed_dim,
                "trans_depth": args.trans_depth,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "image_size": args.image_size,
            },
            "args": vars(args),
        }

        epoch_path = os.path.join(
            SAVE_DIR, f"{run_tag}_epoch{epoch + 1}_auc{val_auc:.4f}.pth")
        torch.save(ckpt_data, epoch_path)
        print(f"  -> Checkpoint saved: {os.path.basename(epoch_path)}")

        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_data["best_auc"] = best_auc
            patience_counter = 0
            best_path = os.path.join(SAVE_DIR, f"{run_tag}_best.pth")
            torch.save(ckpt_data, best_path)
            print(f"  -> New best model saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {args.patience} epochs "
                      f"without improvement.")
                break

        latest_path = os.path.join(SAVE_DIR, f"{run_tag}_latest.pth")
        torch.save(ckpt_data, latest_path)

    print("\n" + "=" * 60)
    print("Evaluating best model on TEST set...")
    best_ckpt = torch.load(
        os.path.join(SAVE_DIR, f"{run_tag}_best.pth"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_auc, test_per_label = evaluate(
        model, test_loader, criterion, device)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Mean AUC: {test_auc:.4f}")
    print("\nPer-label AUC:")
    for label_name, auc_val in test_per_label.items():
        print(f"  {label_name:25s}  {auc_val:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
