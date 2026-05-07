from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from temporal_hl.token_dataset import TemporalHLTokenDataset
from temporal_hl.token_reconstruction_model import TokenReconstructionModel


def masked_l1(pred: torch.Tensor, target: torch.Tensor, joint_valid: torch.Tensor) -> torch.Tensor:
    mask = joint_valid.unsqueeze(-1).repeat(1, 1, 1, 3).reshape(target.shape)
    diff = torch.abs(pred - target) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return diff.sum() / denom


def evaluate(model, loader, device, use_motion: bool) -> dict:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            static_tokens = batch["static_tokens"].to(device)
            motion_tokens = batch["motion_tokens"].to(device) if use_motion else None
            keyframe_mask = batch["keyframe_mask"].to(device) if use_motion else None
            coords = batch["coords"].to(device)
            joint_valid = batch["joint_valid"].to(device)
            pred = model(static_tokens, motion_tokens, keyframe_mask)
            loss = masked_l1(pred, coords, joint_valid)
            losses.append(float(loss.item()))
    return {"coord_l1": sum(losses) / max(len(losses), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train token-to-motion reconstruction.")
    parser.add_argument("--manifest", type=Path, default=Path("temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--mode", choices=["static", "temporal"], default="temporal")
    parser.add_argument("--save-dir", type=Path, default=Path("temporal_hl_cache/runs/token_recon"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_set = TemporalHLTokenDataset(args.manifest, split="train", train=True)
    val_set = TemporalHLTokenDataset(args.manifest, split="train", train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    use_motion = args.mode == "temporal"
    model = TokenReconstructionModel(use_motion=use_motion).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            static_tokens = batch["static_tokens"].to(args.device)
            motion_tokens = batch["motion_tokens"].to(args.device) if use_motion else None
            keyframe_mask = batch["keyframe_mask"].to(args.device) if use_motion else None
            coords = batch["coords"].to(args.device)
            joint_valid = batch["joint_valid"].to(args.device)
            pred = model(static_tokens, motion_tokens, keyframe_mask)
            loss = masked_l1(pred, coords, joint_valid)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
            steps += 1

        metrics = evaluate(model, val_loader, torch.device(args.device), use_motion=use_motion)
        print(
            f"epoch={epoch} mode={args.mode} "
            f"train_coord_l1={running / max(steps, 1):.4f} "
            f"val_coord_l1={metrics['coord_l1']:.4f}"
        )
        if metrics["coord_l1"] < best:
            best = metrics["coord_l1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "metrics": metrics,
                },
                args.save_dir / f"best_{args.mode}.pt",
            )


if __name__ == "__main__":
    main()
