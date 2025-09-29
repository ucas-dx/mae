"""Self-supervised training script for ConvNeXt-Small with configurable stage freezing.

This script loads a ConvNeXt-Small backbone with pretrained weights from timm and
trains it using a SimSiam-style self-supervised objective. The dataset directory
is expected to contain images (optionally arranged in subdirectories). Each image
is augmented twice to produce the positive pairs required by the loss.

Example usage:
    python main_convnext_selfsup.py \
        --data-dir /path/to/images \
        --output-dir ./output \
        --epochs 100 \
        --batch-size 128 \
        --freeze-stages stage0 stage1
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

import timm


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised ConvNeXt-Small trainer")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to store checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Images per batch")
    parser.add_argument("--lr", type=float, default=0.05, help="Base learning rate for SGD")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-stages", nargs="*", default=[],
                        help="Stages to freeze. Choices: stem, stage0, stage1, stage2, stage3")
    parser.add_argument("--projection-dim", type=int, default=2048, help="Projection MLP output dimension")
    parser.add_argument("--prediction-dim", type=int, default=512, help="Prediction MLP hidden dimension")
    parser.add_argument("--print-freq", type=int, default=20, help="Logging frequency in steps")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Checkpoint save frequency in epochs")
    parser.add_argument("--resume", type=str, default="", help="Path to resume checkpoint")
    return parser.parse_args()


class SimpleImageDataset(Dataset):
    """Dataset that loads images from a directory without relying on labels."""

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Data directory {root} does not exist")
        self.samples = [p for p in self.root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
        if not self.samples:
            raise RuntimeError(f"No supported image files found in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image


class TwoCropsTransform:
    """Take two random crops of the same image."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.layer3(x)
        return x


class PredictionMLP(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 2048):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self, backbone: nn.Module, projection_dim: int, prediction_dim: int):
        super().__init__()
        self.backbone = backbone
        feat_dim = getattr(backbone, "num_features", None)
        if feat_dim is None:
            raise AttributeError("Backbone must expose num_features attribute")
        self.projector = ProjectionMLP(feat_dim, hidden_dim=projection_dim, out_dim=projection_dim)
        self.predictor = PredictionMLP(in_dim=projection_dim, hidden_dim=prediction_dim, out_dim=projection_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def simsiam_loss(p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    def negative_cosine(p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return - (p * z).sum(dim=1).mean()

    return 0.5 * (negative_cosine(p1, z2) + negative_cosine(p2, z1))


@dataclass
class AverageMeter:
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def build_backbone(freeze_stages: Sequence[str]) -> nn.Module:
    backbone = timm.create_model("convnext_small", pretrained=True, num_classes=0, global_pool="avg")
    stage_map = {
        "stem": backbone.stem,
        "stage0": backbone.stages[0],
        "stage1": backbone.stages[1],
        "stage2": backbone.stages[2],
        "stage3": backbone.stages[3],
    }
    for stage in freeze_stages:
        if stage not in stage_map:
            raise ValueError(f"Unknown stage '{stage}'. Available: {list(stage_map.keys())}")
        for param in stage_map[stage].parameters():
            param.requires_grad = False
    return backbone


def create_dataloader(args: argparse.Namespace) -> DataLoader:
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    transform = TwoCropsTransform(augmentation)
    dataset = SimpleImageDataset(args.data_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def save_checkpoint(state: dict, is_best: bool, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save(state, path)
    if is_best:
        best_path = output_dir / "model_best.pth"
        torch.save(state, best_path)


def resume_from_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler, path: str) -> int:
    if not path:
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint.get("epoch", 0)
    return start_epoch


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device,
                    epoch: int, print_freq: int) -> float:
    model.train()
    loss_meter = AverageMeter()

    for step, (im1, im2) in enumerate(dataloader):
        im1 = im1.to(device, non_blocking=True)
        im2 = im2.to(device, non_blocking=True)

        optimizer.zero_grad()
        p1, p2, z1, z2 = model(im1, im2)
        loss = simsiam_loss(p1, p2, z1, z2)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), im1.size(0))

        if step % print_freq == 0:
            print(f"Epoch [{epoch}] Step [{step}/{len(dataloader)}] Loss: {loss_meter.val:.4f} (avg: {loss_meter.avg:.4f})")

    return loss_meter.avg


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    dataloader = create_dataloader(args)
    backbone = build_backbone(args.freeze_stages)
    model = SimSiam(backbone, projection_dim=args.projection_dim, prediction_dim=args.prediction_dim)
    model = model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        start_epoch = resume_from_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    output_dir = Path(args.output_dir)
    best_loss = math.inf

    history = {"loss": []}
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, args.print_freq)
        scheduler.step()

        history["loss"].append(avg_loss)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if (epoch + 1) % args.checkpoint_freq == 0 or is_best:
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": avg_loss,
            }
            save_checkpoint(state, is_best, output_dir, f"checkpoint_{epoch + 1:04d}.pth")

        log_path = output_dir / "training_log.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            json.dump(history, f, indent=2)

    print("Training complete. Best loss: {:.4f}".format(best_loss))


if __name__ == "__main__":
    main()
