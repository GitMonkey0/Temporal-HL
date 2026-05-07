from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from temporal_hl.dataset import TemporalHLDataset
from temporal_hl.model import StaticHLBaseline, TemporalHLBaseline


def build_loaders(manifest_path: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_set = TemporalHLDataset(manifest_path=manifest_path, split="train", train=True)
    val_set = TemporalHLDataset(manifest_path=manifest_path, split="train", train=False)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
    )


def compute_loss(
    batch: dict,
    outputs: dict,
    mode: str,
    motion_weight: float = 1.0,
    keyframe_weight: float = 0.5,
    keyframe_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    static_loss = F.cross_entropy(
        outputs["static_logits"].reshape(-1, 26),
        batch["static_tokens"].reshape(-1),
        ignore_index=-100,
    )
    metrics = {"static": float(static_loss.detach().cpu())}
    total = static_loss
    if mode == "temporal":
        motion_loss = F.cross_entropy(
            outputs["motion_logits"].reshape(-1, 27),
            batch["motion_tokens"].reshape(-1),
            ignore_index=-100,
        )
        pos_weight = torch.tensor([keyframe_pos_weight], device=outputs["keyframe_logits"].device)
        keyframe_loss = F.binary_cross_entropy_with_logits(
            outputs["keyframe_logits"],
            batch["keyframe_mask"],
            pos_weight=pos_weight,
        )
        total = total + motion_weight * motion_loss + keyframe_weight * keyframe_loss
        metrics["motion"] = float(motion_loss.detach().cpu())
        metrics["keyframe"] = float(keyframe_loss.detach().cpu())
    return total, metrics


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    correct = (preds == targets) & mask
    denom = mask.sum().item()
    return float(correct.sum().item() / max(denom, 1))


def compute_keyframe_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).int()
    targets = (targets >= 0.5).int()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    motion_weight: float,
    keyframe_weight: float,
    keyframe_pos_weight: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    count = 0
    static_acc = 0.0
    motion_acc = 0.0
    keyframe_f1 = 0.0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            outputs = model(features)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            loss, _ = compute_loss(
                batch,
                outputs,
                mode,
                motion_weight=motion_weight,
                keyframe_weight=keyframe_weight,
                keyframe_pos_weight=keyframe_pos_weight,
            )
            total_loss += float(loss.item())
            count += 1
            static_acc += compute_accuracy(outputs["static_logits"], batch["static_tokens"])
            if mode == "temporal":
                motion_acc += compute_accuracy(outputs["motion_logits"], batch["motion_tokens"])
                keyframe_f1 += compute_keyframe_f1(outputs["keyframe_logits"], batch["keyframe_mask"])
    result = {
        "loss": total_loss / max(count, 1),
        "static_acc": static_acc / max(count, 1),
    }
    if mode == "temporal":
        result["motion_acc"] = motion_acc / max(count, 1)
        result["keyframe_f1"] = keyframe_f1 / max(count, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Temporal Hand Labanotation baseline.")
    parser.add_argument("--manifest", type=Path, default=Path("temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["static", "temporal"], default="temporal")
    parser.add_argument("--save-dir", type=Path, default=Path("temporal_hl_cache/runs/temporal_hl"))
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--keyframe-weight", type=float, default=0.5)
    parser.add_argument("--keyframe-pos-weight", type=float, default=1.0)
    args = parser.parse_args()

    train_loader, val_loader = build_loaders(args.manifest, args.batch_size)
    sample = next(iter(train_loader))
    if args.mode == "temporal":
        model = TemporalHLBaseline(input_dim=sample["features"].shape[-1]).to(args.device)
    else:
        model = StaticHLBaseline(input_dim=sample["features"].shape[-1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            features = batch["features"].to(args.device)
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(features)
            loss, _ = compute_loss(
                batch,
                outputs,
                args.mode,
                motion_weight=args.motion_weight,
                keyframe_weight=args.keyframe_weight,
                keyframe_pos_weight=args.keyframe_pos_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1

        metrics = evaluate(
            model,
            val_loader,
            torch.device(args.device),
            args.mode,
            motion_weight=args.motion_weight,
            keyframe_weight=args.keyframe_weight,
            keyframe_pos_weight=args.keyframe_pos_weight,
        )
        line = (
            f"epoch={epoch} "
            f"mode={args.mode} "
            f"train_loss={running / max(steps, 1):.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_static_acc={metrics['static_acc']:.4f}"
        )
        score = metrics["static_acc"]
        if args.mode == "temporal":
            line += (
                f" val_motion_acc={metrics['motion_acc']:.4f}"
                f" val_keyframe_f1={metrics['keyframe_f1']:.4f}"
            )
            score = 0.5 * (metrics["static_acc"] + metrics["motion_acc"])
        print(line)

        if score > best_metric:
            best_metric = score
            save_path = args.save_dir / f"best_{args.mode}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "metrics": metrics,
                    "input_dim": sample["features"].shape[-1],
                },
                save_path,
            )


if __name__ == "__main__":
    main()
