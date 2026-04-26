from __future__ import annotations

import argparse
from html import parser
import os
import random
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from forge.adapters.qwen_image import QwenImageAdapter
from forge.parallel import FSDPStrategy
from forge.trainer import Trainer
from forge.training_args import TrainingEngineArgs


class SyntheticQwenDataset(Dataset):
    """Produces tokenized inputs shaped for Qwen-Image smoke testing."""

    def __init__(
        self,
        tokenizer: Any,
        length: int = 16,
        image_size: int = 256,   # keep small for dev
        seq_len: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.length = length
        self.image_size = image_size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(index)
        pixel_values = torch.randn((3, self.image_size, self.image_size), generator=gen)
        # Synthetic token ids — replace with real data for actual training
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (self.seq_len,))
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in items]),
        "input_ids": torch.stack([x["input_ids"] for x in items]),
        "attention_mask": torch.stack([x["attention_mask"] for x in items]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", default="outputs/qwen_fsdp")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=10)
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=0)
    parser.add_argument("--validation_every_n_steps", type=int, default=0)
    cli = parser.parse_args()

    args = TrainingEngineArgs(**vars(cli))

    # Init distributed
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")   # different port from SD3 script
    dist.init_process_group(backend=backend)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Build adapter first so we can get the tokenizer for the dataset
    adapter = QwenImageAdapter()
    modules = adapter.build_modules(args)
    tokenizer = modules["runtime_modules"]["tokenizer"]

    dataset = SyntheticQwenDataset(tokenizer=tokenizer)
    train_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    strategy = FSDPStrategy(mixed_precision=args.mixed_precision)

    # Pass already-built adapter so Trainer doesn't call build_modules again
    trainer = Trainer(
        args=args,
        model_adapter=adapter,
        strategy=strategy,
        train_loader=train_loader,
        optimizer_factory=lambda model: torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()