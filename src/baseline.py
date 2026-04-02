"""
Baseline R3D-18 training and evaluation for PrivHAR-Bench.

Config A: Train and evaluate on the same tier (within-tier accuracy).
Config B: Train on Original, evaluate on each privacy tier (cross-domain).
Config C: Train and evaluate on Tier3_AES_B8_NoBG (context-free recognition).

The R3D-18 model is pre-trained on Kinetics-400 and fine-tuned on
PrivHAR-Bench. All hyperparameters, checkpoints, and training logs
are saved for reproducibility.
"""

import json
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .utils import get_class_names, has_valid_json, log, get_device


class FrameSeqDataset(Dataset):
    """Dataset that loads 32-frame PNG sequences for a given tier."""

    def __init__(self, tier_dir, video_ids, class_to_idx, clip_length, resolution):
        self.tier_dir = Path(tier_dir)
        self.video_ids = video_ids
        self.class_to_idx = class_to_idx
        self.clip_length = clip_length
        self.resolution = resolution

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        cls, vstem = self.video_ids[idx]
        label = self.class_to_idx[cls]
        vdir = self.tier_dir / cls / vstem

        frames = []
        for i in range(self.clip_length):
            fpath = vdir / f"frame_{i:04d}.png"
            if fpath.exists():
                img = cv2.imread(str(fpath))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros(
                    (self.resolution[1], self.resolution[0], 3), dtype=np.uint8
                )
            frames.append(img)

        clip = np.stack(frames).astype(np.float32) / 255.0
        clip = clip.transpose(3, 0, 1, 2)  # (C, T, H, W)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
        clip = (clip - mean) / std

        return torch.tensor(clip, dtype=torch.float32), label


def _build_r3d18(num_classes):
    """Build R3D-18 model with Kinetics-400 pre-trained weights."""
    try:
        from torchvision.models.video import R3D_18_Weights, r3d_18
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    except Exception:
        try:
            from torchvision.models.video import r3d_18
            model = r3d_18(pretrained=True)
        except Exception:
            from torchvision.models.video import r3d_18
            model = r3d_18(pretrained=False)
            log("  WARNING: No pretrained weights loaded for R3D-18.")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _train_one_config(tier_name, train_ids, test_ids, class_to_idx, cfg, results_dir):
    """Train R3D-18 on a single tier and return best test accuracy."""
    device = get_device()
    output_dir = Path(cfg["output_dir"])
    resolution = tuple(cfg["resolution"])
    clip_length = cfg["clip_length"]
    train_cfg = cfg["training"]

    ckpt_path = results_dir / f"r3d18_{tier_name}_best.pt"
    result_path = results_dir / f"r3d18_{tier_name}_results.json"

    # Resume check
    if has_valid_json(result_path, min_keys=2):
        r = json.load(open(result_path))
        log(f"  {tier_name}: already done (acc={r['test_acc']:.1f}%)")
        return r

    tier_dir = output_dir / tier_name
    if not tier_dir.exists():
        log(f"  {tier_name}: tier directory not found, skipping")
        return None

    # Verify frames exist
    sample_cls, sample_stem = train_ids[0]
    sample_frame = tier_dir / sample_cls / sample_stem / "frame_0000.png"
    if not sample_frame.exists():
        log(f"  {tier_name}: no frames found ({sample_frame}), skipping")
        return None

    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_workers = train_cfg.get("num_workers", 4)

    train_ds = FrameSeqDataset(tier_dir, train_ids, class_to_idx, clip_length, resolution)
    test_ds = FrameSeqDataset(tier_dir, test_ids, class_to_idx, clip_length, resolution)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    model = _build_r3d18(len(class_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for clips, labels in train_dl:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(clips)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for clips, labels in test_dl:
                clips, labels = clips.to(device), labels.to(device)
                preds = model(clips).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100.0 * correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), str(ckpt_path))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"    {tier_name} epoch {epoch + 1}/{epochs}: "
                f"loss={running_loss / max(len(train_dl), 1):.4f}  "
                f"acc={acc:.1f}%  best={best_acc:.1f}%")

    result = {"tier": tier_name, "test_acc": round(best_acc, 2), "epochs": epochs}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"  {tier_name}: best acc = {best_acc:.1f}%")
    return result


def _evaluate_cross_domain(model_path, eval_tier, test_ids, class_to_idx, cfg):
    """Evaluate a trained model on a different tier (Config B)."""
    device = get_device()
    output_dir = Path(cfg["output_dir"])
    resolution = tuple(cfg["resolution"])
    clip_length = cfg["clip_length"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 4)

    tier_dir = output_dir / eval_tier
    if not tier_dir.exists():
        return None

    model = _build_r3d18(len(class_to_idx)).to(device)
    model.load_state_dict(
        torch.load(str(model_path), map_location=device, weights_only=True)
    )
    model.eval()

    test_ds = FrameSeqDataset(tier_dir, test_ids, class_to_idx, clip_length, resolution)
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    correct, total = 0, 0
    with torch.no_grad():
        for clips, labels in test_dl:
            clips, labels = clips.to(device), labels.to(device)
            preds = model(clips).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return round(100.0 * correct / max(total, 1), 2)


def run_baseline_training(cfg):
    """
    Phase 6: Train R3D-18 baselines under Config A and Config B.

    Args:
        cfg: Loaded YAML configuration dict.

    Returns:
        Dict with config_a and config_b results.
    """
    log("=== PHASE 6: BASELINE TRAINING ===")

    device = get_device()
    if device == "cpu":
        log("  WARNING: No CUDA GPU detected. Training on CPU is very slow.")

    output_dir = Path(cfg["output_dir"])
    classes = get_class_names(cfg)
    results_dir = output_dir / "baseline_results"
    results_dir.mkdir(exist_ok=True)

    final_path = results_dir / "all_results.json"
    if has_valid_json(final_path, min_keys=2):
        data = json.load(open(final_path))
        if data.get("config_a") and len(data["config_a"]) >= 5:
            log("  Already completed with valid data. Skipping.")
            return data

    if final_path.exists():
        log("  Found stale/empty all_results.json; recomputing.")
        final_path.unlink()

    ann_path = output_dir / "annotations.json"
    if not ann_path.exists():
        log("  annotations.json not found. Run metadata generation first.")
        return None

    annotations = json.load(open(ann_path))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_ids = [
        (a["class"], a["source_file"].replace(".avi", ""))
        for a in annotations
        if a["split"] == "train" and a["class"] in classes
    ]
    test_ids = [
        (a["class"], a["source_file"].replace(".avi", ""))
        for a in annotations
        if a["split"] == "test" and a["class"] in classes
    ]

    log(f"  Classes: {len(classes)}, Train: {len(train_ids)}, Test: {len(test_ids)}")

    config_a_tiers = [
        "Original", "Tier1_Blur", "Tier2_Edge",
        "Tier3_AES_B4", "Tier3_AES_B8", "Tier3_AES_B16",
        "Tier3_AES_B8_NoBG",
    ]

    # Config A: train and evaluate on the same tier
    config_a = {}
    for tier in config_a_tiers:
        try:
            r = _train_one_config(tier, train_ids, test_ids, class_to_idx, cfg, results_dir)
            if r:
                config_a[tier] = r["test_acc"]
        except Exception as e:
            log(f"  ERROR training {tier}: {e}")
            traceback.print_exc()

    # Config B: train on Original, evaluate on each other tier
    config_b = {}
    orig_ckpt = results_dir / "r3d18_Original_best.pt"
    if orig_ckpt.exists():
        log("  Config B: Evaluating Original model on each tier...")
        for tier in config_a_tiers:
            if tier == "Original":
                continue
            try:
                acc = _evaluate_cross_domain(
                    orig_ckpt, tier, test_ids, class_to_idx, cfg
                )
                if acc is not None:
                    config_b[tier] = acc
                    log(f"    Config B {tier}: {acc:.1f}%")
            except Exception as e:
                log(f"  ERROR Config B {tier}: {e}")

    all_results = {"config_a": config_a, "config_b": config_b}
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log("  All training results saved.")

    return all_results
