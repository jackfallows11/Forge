from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from forge.parallel.base import ParallelStrategy

try:
    import torch.distributed as dist
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint import save as dcp_save
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
except Exception as exc:  # pragma: no cover - import guard for local environments
    dist = None
    FSDP = None
    _FSDP_IMPORT_ERROR = exc
else:
    _FSDP_IMPORT_ERROR = None


class FSDPStrategy(ParallelStrategy):
    def __init__(self, mixed_precision: str = "bf16") -> None:
        self.mixed_precision = mixed_precision
        self.model: Any | None = None
        self.optimizer: Any | None = None

    def prepare_model(self, model: Any) -> Any:
        if FSDP is None or dist is None:
            raise RuntimeError("FSDP is unavailable in this environment") from _FSDP_IMPORT_ERROR
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before FSDPStrategy.prepare_model")

        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        # model.to(device)
        mp_policy = self._build_mixed_precision()
        is_meta = any(p.device.type == "meta" for p in model.parameters())

        def param_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=device)
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.model = FSDP(
            model,
            device_id=device,
            use_orig_params=True,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=not is_meta,
            param_init_fn=param_init_fn if is_meta else None,
        )
        return self.model

    def prepare_optimizer(self, model: Any, optimizer: Any) -> Any:
        del model
        self.optimizer = optimizer
        return optimizer

    def backward(self, loss: Any) -> None:
        loss.backward()

    def step(self, optimizer: Any, scheduler: Any | None = None) -> None:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    def clip_grad_norm_(self, model: Any, max_norm: float) -> None:
        if max_norm > 0:
            model.clip_grad_norm_(max_norm)

    def save(self, path: str, state: dict[str, Any]) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been prepared")
        ckpt_dir = Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state_dict = {
            "model": get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            ),
            "optimizer": get_optimizer_state_dict(
                self.model,
                self.optimizer,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            ) if self.optimizer is not None else {},
            "meta": state,
        }
        dcp_save(state_dict=state_dict, checkpoint_id=ckpt_dir)

    def load(self, path: str, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
        state_dict = {
            "model": get_model_state_dict(
                model,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            ),
            "optimizer": get_optimizer_state_dict(
                model,
                optimizer,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            ) if optimizer is not None else {},
            "meta": {},
        }
        dcp_load(state_dict=state_dict, checkpoint_id=Path(path))
        set_model_state_dict(model, model_state_dict=state_dict["model"])
        if optimizer is not None:
            set_optimizer_state_dict(
                model,
                optimizer,
                optim_state_dict=state_dict["optimizer"],
            )
        return state_dict["meta"]

    def _build_mixed_precision(self) -> Any:
        if self.mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif self.mixed_precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
