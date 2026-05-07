from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from temporal_hl.seqname_dataset import SeqNameTokenDataset
from temporal_hl.token_action_model import TokenActionClassifier


def evaluate(model, loader, device, use_motion: bool, use_keyframe: bool) -> dict:
    model.eval()
    total = 0
    correct = 0
    losses = []
    with torch.no_grad():
        for batch in loader:
            static_tokens = batch["static_tokens"].to(device)
            motion_tokens = batch["motion_tokens"].to(device) if use_motion else None
            keyframe_mask = batch["keyframe_mask"].to(device) if use_keyframe else None
            labels = batch["label"].to(device)
            logits = model(static_tokens, motion_tokens, keyframe_mask)
            loss = F.cross_entropy(logits, labels)
            losses.append(float(loss.item()))
            pred = logits.argmax(dim=-1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    return {"loss": sum(losses) / max(len(losses), 1), "acc": correct / max(total, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action classifier from notation tokens.")
    parser.add_argument("--manifest", type=Path, default=Path("temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--mode", choices=["static", "keyframe", "temporal"], default="temporal")
    parser.add_argument("--save-dir", type=Path, default=Path("temporal_hl_cache/runs/token_action"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_set = SeqNameTokenDataset(args.manifest, args.data_root, split="train", mode=args.mode)
    test_set = SeqNameTokenDataset(args.manifest, args.data_root, split="test", mode=args.mode)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    use_motion = args.mode == "temporal"
    use_keyframe = args.mode in ("keyframe", "temporal")
    model = TokenActionClassifier(
        num_classes=len(train_set.label_to_idx),
        use_motion=use_motion,
        use_keyframe=use_keyframe,
    ).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            static_tokens = batch["static_tokens"].to(args.device)
            motion_tokens = batch["motion_tokens"].to(args.device) if use_motion else None
            keyframe_mask = batch["keyframe_mask"].to(args.device) if use_keyframe else None
            labels = batch["label"].to(args.device)
            logits = model(static_tokens, motion_tokens, keyframe_mask)
            loss = F.cross_entropy(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
            steps += 1
        metrics = evaluate(model, test_loader, torch.device(args.device), use_motion, use_keyframe)
        print(
            f"epoch={epoch} mode={args.mode} "
            f"train_loss={running / max(steps,1):.4f} "
            f"test_loss={metrics['loss']:.4f} test_acc={metrics['acc']:.4f}"
        )
        if metrics["acc"] > best:
            best = metrics["acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "metrics": metrics,
                    "label_to_idx": train_set.label_to_idx,
                },
                args.save_dir / f"best_{args.mode}.pt",
            )


if __name__ == "__main__":
    main()
