"""Train CFRNet with multi-scale SLS supervision."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfrnet import CFRNet
from cfrnet.data import IRSTDDataset, canonical_dataset_name, default_batch_size, default_input_size
from cfrnet.loss import MultiScaleSLSLoss
from cfrnet.metrics import AverageMeter, SegMetrics, scalar_metrics
from cfrnet.seed import seed_everything, seed_worker


def parse_args():
    parser = argparse.ArgumentParser(description="Train CFRNet")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--dataset", type=canonical_dataset_name, default="NUAA-SIRST")
    parser.add_argument("--split-root", type=Path, default=Path(__file__).parent / "splits")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-start", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("runs/CFRNet"))
    parser.add_argument("--resume", type=Path, help="Resume a trusted last.pth checkpoint")
    args = parser.parse_args()
    if args.epochs < 1 or args.eval_start < 1 or args.eval_every < 1:
        parser.error("epochs, eval-start and eval-every must be positive")
    if args.workers < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("workers must be nonnegative and batch-size must be positive")
    if not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        parser.error("Require 0 <= min-lr <= lr and lr > 0")
    if not 0 < args.threshold < 1:
        parser.error("threshold must be in (0, 1)")
    args.batch_size = args.batch_size or default_batch_size(args.dataset)
    return args


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, threshold):
    model.eval()
    losses = AverageMeter()
    metrics = SegMetrics(
        threshold=threshold, match_distance=3.0, roc_bins=200, pd_match_mode="centroid"
    )
    for images, masks, _ in tqdm(loader, desc="Evaluate", leave=False):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images, return_aux=True)
        loss = criterion(outputs, masks, epoch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite evaluation loss")
        losses.update(loss.item(), images.shape[0])
        metrics.update(outputs["pred"].sigmoid(), masks)
    result = scalar_metrics(metrics.get())
    result["loss"] = losses.avg
    return result


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke check")
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            "Output directory is not empty; choose a new --output or use --resume"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    size = default_input_size(args.dataset)
    train_set = IRSTDDataset(
        args.dataset_root, args.dataset, "train", size=size, split_root=args.split_root
    )
    test_set = IRSTDDataset(
        args.dataset_root, args.dataset, "test", size=size, split_root=args.split_root
    )
    if not len(train_set) or not len(test_set):
        raise ValueError("Training and test splits must be nonempty")
    if set(train_set.ids) & set(test_set.ids):
        raise ValueError("Training and test splits overlap")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set, batch_size=1, num_workers=args.workers, worker_init_fn=seed_worker
    )
    model = CFRNet().to(device)
    criterion = MultiScaleSLSLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    start_epoch, best_f1 = 0, -1.0
    if args.resume:
        # Only load checkpoints you trust: torch.load uses Python pickle.
        checkpoint = torch.load(args.resume, map_location="cpu")
        for key in (
            "dataset",
            "dataset_root",
            "split_root",
            "epochs",
            "batch_size",
            "seed",
            "lr",
            "min_lr",
            "eval_start",
            "eval_every",
            "threshold",
            "workers",
        ):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Resume configuration differs for {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_epoch, best_f1 = checkpoint["epoch"], checkpoint["best_f1"]
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.set_rng_state(checkpoint["torch_state"])
        generator.set_state(checkpoint["loader_state"])
        if device.type == "cuda" and checkpoint["cuda_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_state"])
    if start_epoch >= args.epochs:
        raise ValueError("Checkpoint has already reached the requested epoch count")
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = args.output / "history.csv"
    fields = [
        "epoch",
        "lr",
        "train_loss",
        "eval_loss",
        "IoU",
        "nIoU",
        "F1",
        "Precision",
        "Recall",
        "PD",
        "FA_x1e6",
    ]
    with log_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        for epoch_index in range(start_epoch, args.epochs):
            epoch = epoch_index + 1
            # Match the original cosine convention, extended to 1000 epochs.
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
                1 + math.cos(math.pi * epoch / args.epochs)
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            losses = AverageMeter()
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            for images, masks, _ in progress:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                # Original first epoch trains the finest head before fused supervision.
                outputs = model(images, return_aux=True, warmup=epoch_index == 0)
                loss = criterion(outputs, masks, epoch_index)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), images.shape[0])
                progress.set_postfix(loss=f"{losses.avg:.5f}", lr=f"{lr:.3g}")
            should_eval = (
                epoch >= args.eval_start and (epoch - args.eval_start) % args.eval_every == 0
            ) or epoch == args.epochs
            result = (
                evaluate(model, test_loader, criterion, device, epoch, args.threshold)
                if should_eval
                else None
            )
            improved = result is not None and result["F1"] > best_f1
            if improved:
                best_f1 = result["F1"]
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": config,
                "random_state": random.getstate(),
                "numpy_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
                "loader_state": generator.get_state(),
                "cuda_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }
            save_checkpoint(args.output / "last.pth", state)
            if improved:
                save_checkpoint(args.output / "best.pth", state)
            row = {"epoch": epoch, "lr": lr, "train_loss": losses.avg}
            if result:
                row.update({key: result[key] for key in fields if key in result})
                row["eval_loss"] = result["loss"]
                print(
                    f"Epoch {epoch}: IoU={result['IoU']:.4f}, F1={result['F1']:.4f}, best F1={best_f1:.4f}"
                )
            writer.writerow(row)
            handle.flush()


if __name__ == "__main__":
    main()
