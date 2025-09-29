"""Self-supervised training entry point for ConvNeXt-Small using a SimSiam objective."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import timm

import util.lr_sched as lr_sched
import util.misc as misc


FREEZEABLE_STAGES = ("stem", "stage0", "stage1", "stage2", "stage3")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class UnlabeledImageDataset(Dataset):
    """Iterates all images under a root directory without requiring class labels."""

    def __init__(self, root: str, transform=None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Data directory {root} does not exist")

        self.transform = transform
        self.samples = [p for p in sorted(self.root.rglob("*")) if p.suffix.lower() in SUPPORTED_EXTS]
        if not self.samples:
            raise RuntimeError(f"No supported image files were found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):  # type: ignore[override]
        path = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image


class TwoCropsTransform:
    """Apply the same base transform twice to create positive pairs."""

    def __init__(self, base_transform) -> None:
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.layer3(x)
        return x


class PredictionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self, backbone: nn.Module, projection_dim: int, prediction_dim: int) -> None:
        super().__init__()
        if not hasattr(backbone, "num_features"):
            raise AttributeError("Backbone must expose a 'num_features' attribute")
        feat_dim = int(backbone.num_features)  # type: ignore[attr-defined]
        self.backbone = backbone
        self.projector = ProjectionMLP(feat_dim, projection_dim, projection_dim)
        self.predictor = PredictionMLP(projection_dim, prediction_dim, projection_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):  # noqa: D401
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def simsiam_loss(p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    def negative_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    return 0.5 * (negative_cosine(p1, z2) + negative_cosine(p2, z1))


def build_backbone(freeze_stages: Sequence[str]) -> nn.Module:
    backbone = timm.create_model(
        "convnext_small",
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    )

    freeze_map = _collect_convnext_stage_modules(backbone)
    unknown = set(freeze_stages) - set(freeze_map.keys())
    if unknown:
        raise ValueError(f"Unknown stages requested to freeze: {sorted(unknown)}")

    for stage in freeze_stages:
        for module in freeze_map[stage]:
            for param in module.parameters():
                param.requires_grad = False

    return backbone


def _collect_convnext_stage_modules(backbone: nn.Module) -> Dict[str, List[nn.Module]]:
    stage_modules: Dict[str, List[nn.Module]] = {stage: [] for stage in FREEZEABLE_STAGES}

    if hasattr(backbone, "stem"):
        stage_modules["stem"].append(backbone.stem)  # type: ignore[attr-defined]

    if hasattr(backbone, "stages"):
        stages = list(backbone.stages)  # type: ignore[attr-defined]
        for idx, stage in enumerate(stages):
            stage_key = f"stage{idx}"
            if stage_key in stage_modules:
                stage_modules[stage_key].append(stage)

    if hasattr(backbone, "downsample_layers"):
        downsample_layers = list(backbone.downsample_layers)  # type: ignore[attr-defined]
        if downsample_layers:
            stage_modules["stem"].append(downsample_layers[0])
            for idx, layer in enumerate(downsample_layers[1:]):
                stage_key = f"stage{idx}"
                if stage_key in stage_modules:
                    stage_modules[stage_key].append(layer)

    return {k: v for k, v in stage_modules.items() if v}


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("ConvNeXt-Small self-supervised training", add_help=False)
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size per process")
    parser.add_argument("--epochs", default=100, type=int, help="Total training epochs")
    parser.add_argument("--accum_iter", default=1, type=int, help="Gradient accumulation steps")

    parser.add_argument("--lr", default=None, type=float, help="Absolute learning rate")
    parser.add_argument("--blr", default=0.05, type=float, help="Base learning rate")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="Lower learning rate bound")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Warmup epochs for LR schedule")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")

    parser.add_argument("--data_path", required=True, type=str, help="Directory with training images")
    parser.add_argument("--output_dir", default="./output_dir", type=str, help="Where to store checkpoints")
    parser.add_argument("--log_dir", default="./output_dir", type=str, help="Where to store training logs")
    parser.add_argument("--device", default="cuda", type=str, help="Training device")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--resume", default="", type=str, help="Checkpoint path to resume from")
    parser.add_argument("--start_epoch", default=0, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--num_workers", default=8, type=int, help="Data loading workers")
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--print_freq", default=50, type=int, help="Logging frequency")
    parser.add_argument(
        "--freeze_stages",
        default=[],
        nargs="*",
        choices=FREEZEABLE_STAGES,
        help="Backbone stages to freeze during training",
    )
    parser.add_argument("--projection_dim", default=2048, type=int, help="Projection head dimension")
    parser.add_argument("--prediction_dim", default=512, type=int, help="Prediction head hidden dimension")

    parser.add_argument("--world_size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="URL used to set up distributed training")

    return parser


def main(args) -> None:
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print(str(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Data pipeline
    base_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_train = UnlabeledImageDataset(args.data_path, transform=TwoCropsTransform(base_transform))

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Model + optimizer
    backbone = build_backbone(args.freeze_stages)
    model = SimSiam(backbone, args.projection_dim, args.prediction_dim)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    optimizer = optim.SGD(
        (p for p in model_without_ddp.parameters() if p.requires_grad),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    misc.load_model(args, model_without_ddp, optimizer, loss_scaler=None)

    print(f"Start training for {args.epochs} epochs")
    start_time = datetime.datetime.now().timestamp()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        lr_sched.adjust_learning_rate(optimizer, epoch, args)
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, args)

        if args.output_dir:
            checkpoint_path = output_dir / f"checkpoint-{epoch:04d}.pth"
            misc.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                },
                checkpoint_path,
            )

        log_stats = {**{k: float(v) for k, v in train_stats.items()}, "epoch": epoch}
        if args.output_dir and misc.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = datetime.datetime.now().timestamp() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
) -> Dict[str, float]:
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"

    optimizer.zero_grad()
    steps = 0
    for data_iter_step, (views_one, views_two) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images1 = views_one.to(device, non_blocking=True)
        images2 = views_two.to(device, non_blocking=True)

        p1, p2, z1, z2 = model(images1, images2)
        loss = simsiam_loss(p1, p2, z1, z2)
        loss_value = loss.item()
        loss = loss / args.accum_iter

        loss.backward()
        steps += 1
        if steps % args.accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])

    if steps % args.accum_iter != 0:
        optimizer.step()
        optimizer.zero_grad()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ConvNeXt self-supervised training", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
