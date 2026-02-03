import os
import argparse
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset import NIHChestXrayDataset, LABELS
from model import LTViT

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "datasets", "nih-chest-xrays", "data", "versions", "3",
)
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "checkpoints", "lt_vit_best.pth",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LT-ViT")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image for inference")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for positive prediction")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    model = LTViT(
        num_labels=len(LABELS),
        n2=saved_args.get("n2", 4),
        pretrained=False,
        image_size=saved_args.get("image_size", 224),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    print(f"  Trained for {ckpt['epoch'] + 1} epochs, best AUC: {ckpt['best_auc']:.4f}")
    return model, saved_args


def predict_single_image(model, image_path, device, image_size=224, threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad(), autocast(device.type):
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    print(f"\nPredictions for: {image_path}")
    print("-" * 45)
    for label_name, prob in zip(LABELS, probs):
        marker = " *" if prob >= threshold else ""
        print(f"  {label_name:25s}  {prob:.4f}{marker}")
    print("-" * 45)
    print("  (* = above threshold)")

    return dict(zip(LABELS, probs))


def evaluate_test_set(model, device, batch_size, num_workers, threshold=0.5):
    test_set = NIHChestXrayDataset(DATA_DIR, split="test", image_size=224)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"\nEvaluating on {len(test_set)} test images...")

    all_labels = []
    all_probs = []

    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), autocast(device.type):
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    print("\n" + "=" * 55)
    print(f"{'Label':25s}  {'AUC-ROC':>8s}  {'Prevalence':>10s}")
    print("-" * 55)
    aucs = []
    for i, label_name in enumerate(LABELS):
        prevalence = all_labels[:, i].mean()
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            aucs.append(auc)
        else:
            auc = float("nan")
        print(f"  {label_name:25s}  {auc:8.4f}  {prevalence:10.4f}")

    mean_auc = np.mean(aucs) if aucs else 0.0
    print("-" * 55)
    print(f"  {'Mean AUC':25s}  {mean_auc:8.4f}")
    print("=" * 55)

    all_preds = (all_probs >= threshold).astype(int)
    print(f"\nClassification report (threshold={threshold}):\n")
    print(classification_report(all_labels, all_preds, target_names=LABELS,
                                zero_division=0))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, saved_args = load_model(args.checkpoint, device)

    if args.image:
        predict_single_image(
            model, args.image, device,
            image_size=saved_args.get("image_size", 224),
            threshold=args.threshold,
        )
    else:
        evaluate_test_set(
            model, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
