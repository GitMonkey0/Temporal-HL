from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from temporal_hl.token_dataset import TemporalHLTokenDataset
from temporal_hl.token_reconstruction_baselines import GRUTokenReconstructionModel, TCNTokenReconstructionModel
from temporal_hl.token_reconstruction_model import TokenReconstructionModel


def masked_l1(pred: torch.Tensor, target: torch.Tensor, joint_valid: torch.Tensor) -> torch.Tensor:
    mask = joint_valid.unsqueeze(-1).repeat(1, 1, 1, 3).reshape(target.shape)
    diff = torch.abs(pred - target) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return diff.sum() / denom


def build_model(
    arch: str,
    use_motion: bool,
    use_keyframe: bool,
    d_model: int,
    num_layers: int,
    nhead: int,
    dropout: float,
):
    if arch == "transformer":
        return TokenReconstructionModel(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_motion=use_motion,
            use_keyframe=use_keyframe,
        )
    if arch == "gru":
        return GRUTokenReconstructionModel(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            use_motion=use_motion,
            use_keyframe=use_keyframe,
        )
    if arch == "tcn":
        return TCNTokenReconstructionModel(
            d_model=d_model,
            depth=num_layers,
            dropout=dropout,
            use_motion=use_motion,
            use_keyframe=use_keyframe,
        )
    raise ValueError(f"Unknown arch: {arch}")


def evaluate(model, loader, device, use_motion: bool, use_keyframe: bool) -> dict:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            static_tokens = batch["static_tokens"].to(device)
            motion_tokens = batch["motion_tokens"].to(device) if use_motion else None
            keyframe_mask = batch["keyframe_mask"].to(device) if use_keyframe else None
            coords = batch["coords"].to(device)
            joint_valid = batch["joint_valid"].to(device)
            pred = model(static_tokens, motion_tokens, keyframe_mask)
            loss = masked_l1(pred, coords, joint_valid)
            losses.append(float(loss.item()))
    return {"coord_l1": sum(losses) / max(len(losses), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stronger token-to-motion baselines.")
    parser.add_argument("--manifest", type=Path, default=Path("temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--mode", choices=["static", "motion", "keyframe", "temporal"], default="temporal")
    parser.add_argument("--arch", choices=["transformer", "gru", "tcn"], default="gru")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=Path("temporal_hl_cache/runs/token_recon_v2"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_set = TemporalHLTokenDataset(args.manifest, split="train", train=True)
    val_set = TemporalHLTokenDataset(args.manifest, split="train", train=False)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    use_motion = args.mode in ("motion", "temporal")
    use_keyframe = args.mode in ("keyframe", "temporal")
    model = build_model(
        args.arch,
        use_motion=use_motion,
        use_keyframe=use_keyframe,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
    ).to(args.device)
    if args.init_from is not None and args.init_from.exists():
        ckpt = torch.load(args.init_from, map_location="cpu")
        init_state = ckpt["model_state"]
        model_state = model.state_dict()
        compatible = {
            k: v
            for k, v in init_state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        model_state.update(compatible)
        model.load_state_dict(model_state)
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
            keyframe_mask = batch["keyframe_mask"].to(args.device) if use_keyframe else None
            coords = batch["coords"].to(args.device)
            joint_valid = batch["joint_valid"].to(args.device)
            pred = model(static_tokens, motion_tokens, keyframe_mask)
            loss = masked_l1(pred, coords, joint_valid)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
            steps += 1

        metrics = evaluate(
            model,
            val_loader,
            torch.device(args.device),
            use_motion=use_motion,
            use_keyframe=use_keyframe,
        )
        print(
            f"epoch={epoch} arch={args.arch} mode={args.mode} "
            f"train_coord_l1={running / max(steps, 1):.4f} "
            f"val_coord_l1={metrics['coord_l1']:.4f}"
        )
        if metrics["coord_l1"] < best:
            best = metrics["coord_l1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "arch": args.arch,
                    "d_model": args.d_model,
                    "num_layers": args.num_layers,
                    "nhead": args.nhead,
                    "dropout": args.dropout,
                    "seed": args.seed,
                    "metrics": metrics,
                },
                args.save_dir / f"best_{args.arch}_{args.mode}.pt",
            )


if __name__ == "__main__":
    main()
