from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from forge.adapters.base import ModelAdapter
from forge.state import BatchState

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
except Exception as exc:
    AutoConfig = AutoModel = AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class QwenImageAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.model: Any | None = None
        self.train_model: Any | None = None
        self.runtime_modules: dict[str, Any] = {}

    def build_modules(self, args: Any) -> dict[str, Any]:
        if self.model is not None:
            return {
                "model": self.model,
                "runtime_modules": self.runtime_modules,
                "trainable_modules": ["model"],
            }
        
        if AutoConfig is None:
            raise RuntimeError("transformers is unavailable") from _TRANSFORMERS_IMPORT_ERROR

        # Load config only — no weights yet, no memory used
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )

        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        model.requires_grad_(True)

        self.runtime_modules["model"] = model

        # Tokenizer is small, fine to load normally
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        self.runtime_modules["tokenizer"] = tokenizer

        return {
            "model": model,
            "runtime_modules": self.runtime_modules,
            "trainable_modules": ["model"],
        }

    def prepare_batch(self, raw_batch: dict[str, Any]) -> BatchState:
        return BatchState(raw_batch=raw_batch)

    def set_train_model(self, model: Any) -> None:
        # Called after FSDP wraps the model
        self.train_model = model
        self.runtime_modules["model"] = model

    def forward_loss(self, batch_state: BatchState) -> dict[str, Any]:
        model = self.train_model
        if model is None:
            raise RuntimeError("build_modules must be called before forward_loss")

        raw = batch_state.raw_batch
        device = next(model.parameters()).device
        dtype = torch.bfloat16  # match what you loaded with

        # Map your dataloader keys to what Qwen-Image's forward actually expects.
        # Check the Qwen-Image model card / source for the exact argument names.
        pixel_values = raw["pixel_values"].to(device=device, dtype=dtype)
        input_ids = raw["input_ids"].to(device=device)
        attention_mask = raw.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,   # for autoregressive loss; adjust if Qwen-Image differs
        )
        loss = outputs.loss

        batch_state.loss_dict["loss"] = loss
        batch_state.metrics["loss"] = float(loss.detach().item())
        return {
            "loss": loss,
            "metrics": batch_state.metrics,
            "artifacts": batch_state.artifacts,
        }

    def validation_generate(self, prompts: list[str], output_dir: str) -> dict[str, Any]:
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return {"sample_paths": []}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Placeholder — fill in Qwen-Image generation call when ready
        return {"sample_paths": []}