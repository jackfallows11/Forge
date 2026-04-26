from __future__ import annotations

import argparse
import os
import random
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from forge.adapters import DiffusersSD3Adapter
from forge.parallel import FSDPStrategy
from forge.trainer import Trainer
from forge.training_args import TrainingEngineArgs

import bitsandbytes as bnb

class SyntheticTextImageDataset(Dataset):
    def __init__(self, length: int = 16, image_size: int = 256) -> None:
        self.length = length
        self.image_size = image_size
        self.prompts = [
            "a clean poster that says FORGE",
            "a printed receipt with legible OCR-friendly text",
            "a bookstore sign that says OPEN TODAY",
        ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(index)
        pixel_values = torch.randn((3, self.image_size, self.image_size), generator=gen)
        prompt = self.prompts[index % len(self.prompts)]
        return {"pixel_values": pixel_values, "prompts": prompt}


def collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in items], dim=0),
        "prompts": [item["prompts"] for item in items],
    }


def build_optimizer(model: torch.nn.Module, args: TrainingEngineArgs) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    return bnb.optim.AdamW8bit(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def parse_args() -> TrainingEngineArgs:
    parser = argparse.ArgumentParser(description="M0 SD3 T2I FSDP trainer")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", default="outputs/m0_sd3_fsdp")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_train_steps", type=int, default=10)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=5)
    parser.add_argument("--validation_every_n_steps", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)
    cli_args = parser.parse_args()
    return TrainingEngineArgs(**vars(cli_args))


def init_dist() -> None:
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    init_dist()
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dataset = SyntheticTextImageDataset()
    train_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    adapter = DiffusersSD3Adapter()
    strategy = FSDPStrategy(mixed_precision=args.mixed_precision)
    trainer = Trainer(
        args=args,
        model_adapter=adapter,
        strategy=strategy,
        train_loader=train_loader,
        optimizer_factory=lambda model: build_optimizer(model, args),
    )
    trainer.train()


if __name__ == "__main__":
    main()
