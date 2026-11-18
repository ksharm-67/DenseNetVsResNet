import argparse
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CLASSES, DISEASE_IDX, CustomImageDataset, build_splits
from model import ARCHS, define_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=ARCHS, default="resnet50")
    p.add_argument("--data_root", default="/kaggle/input/data",
                   help="Folder containing Data_Entry_2017.csv, the list txt files and images_*/")
    p.add_argument("--out_dir", default="/kaggle/working/runs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0, help="Training seed (init of head, shuffling)")
    p.add_argument("--split_seed", type=int, default=42, help="Seed for the patient-level train/val split")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--subset", type=int, default=None, help="Use N images per split (quick smoke test)")
    p.add_argument("--no_pretrained", action="store_true", help="Skip downloading ImageNet weights (testing only)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_class_auc(y, p):
    aucs = []
    for i in range(y.shape[1]):
        #AUC undefined if a class has one label value
        aucs.append(float(roc_auc_score(y[:, i], p[:, i])) if 0 < y[:, i].sum() < len(y) else float("nan"))
    return aucs


def mean_disease_auc(aucs):
    return float(np.nanmean([aucs[i] for i in DISEASE_IDX]))


@torch.no_grad()
def predict(model, dl, criterion, device, use_amp):
    model.eval()
    total_loss, probs, labels = 0.0, [], []
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
        out = out.float()
        total_loss += criterion(out, y).item() * x.size(0)
        probs.append(torch.sigmoid(out).cpu())
        labels.append(y.cpu())
    return total_loss / len(dl.dataset), torch.cat(labels).numpy(), torch.cat(probs).numpy()


def train_one_epoch(model, dl, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running = 0.0
    for i, (x, y) in enumerate(dl):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
        loss = criterion(out.float(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * x.size(0)
        if i % 100 == 0:
            print(f"  batch {i}/{len(dl)}  loss {loss.item():.4f}", flush=True)
    return running / len(dl.dataset)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    run_name = f"{args.arch}_seed{args.seed}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run {run_name} on {device}", flush=True)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        norm,
    ])

    splits = build_splits(args.data_root, split_seed=args.split_seed)
    if args.subset:
        splits = {k: (v[0][: args.subset], v[1][: args.subset]) for k, v in splits.items()}
    print({k: len(v[0]) for k, v in splits.items()}, flush=True)

    ds = {
        "train": CustomImageDataset(*splits["train"], transform=train_tf, cache_dir=args.cache_dir),
        "val": CustomImageDataset(*splits["val"], transform=eval_tf, cache_dir=args.cache_dir),
        "test": CustomImageDataset(*splits["test"], transform=eval_tf, cache_dir=args.cache_dir),
    }
    gen = torch.Generator().manual_seed(args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=use_amp,
                  persistent_workers=args.workers > 0)
    train_dl = DataLoader(ds["train"], shuffle=True, generator=gen, drop_last=True, **common)
    val_dl = DataLoader(ds["val"], shuffle=False, **common)
    test_dl = DataLoader(ds["test"], shuffle=False, **common)

    model = define_model(args.arch, num_classes=len(CLASSES), pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    history, best_auc, best_epoch = [], -1.0, -1
    ckpt_path = os.path.join(run_dir, "best.pth")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, scaler, device, use_amp)
        val_loss, yv, pv = predict(model, val_dl, criterion, device, use_amp)
        val_auc = mean_disease_auc(per_class_auc(yv, pv))
        secs = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_mean_auc": val_auc, "seconds": secs})
        print(f"Epoch {epoch}: train {train_loss:.4f}  val {val_loss:.4f}  "
              f"val mean AUC {val_auc:.4f}  ({secs:.0f}s)", flush=True)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), ckpt_path)

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if use_amp else None

    #Test with best val checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, yt, pt = predict(model, test_dl, criterion, device, use_amp)
    aucs = per_class_auc(yt, pt)
    test_mean = mean_disease_auc(aucs)
    print(f"Test mean AUC (14 diseases, epoch {best_epoch} weights): {test_mean:.4f}")
    for c, a in zip(CLASSES, aucs):
        print(f"  {c:20s} {a:.4f}")

    np.savez_compressed(os.path.join(run_dir, "test_preds.npz"), labels=yt, probs=pt)

    plt.figure(figsize=(7, 6))
    for i in DISEASE_IDX:
        if np.isnan(aucs[i]):
            continue
        fpr, tpr, _ = roc_curve(yt[:, i], pt[:, i])
        plt.plot(fpr, tpr, label=f"{CLASSES[i]} ({aucs[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves, {args.arch} (seed {args.seed})")
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "roc.png"), dpi=150)
    plt.close()

    results = {
        "arch": args.arch, "seed": args.seed, "args": vars(args),
        "n_params": n_params, "peak_gpu_mem_gb": peak_mem_gb,
        "best_epoch": best_epoch, "best_val_mean_auc": best_auc,
        "test_loss": test_loss, "test_mean_auc": test_mean,
        "test_auc_per_class": dict(zip(CLASSES, aucs)),
        "history": history,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
