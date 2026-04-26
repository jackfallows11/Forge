from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from forge.adapters.base import ModelAdapter
from forge.state import BatchState

try:
    from diffusers import StableDiffusion3Pipeline
except Exception as exc:  # pragma: no cover - import guard for local environments
    StableDiffusion3Pipeline = None
    _DIFFUSERS_IMPORT_ERROR = exc
else:
    _DIFFUSERS_IMPORT_ERROR = None


class DiffusersSD3Adapter(ModelAdapter):
    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.train_model: Any | None = None
        self.runtime_modules: dict[str, Any] = {}
        self.trainable_modules: list[str] = []
        self.primary_train_model: Any | None = None

    def build_modules(self, args: Any) -> dict[str, Any]:
        if StableDiffusion3Pipeline is None:
            raise RuntimeError("diffusers is unavailable in this environment") from _DIFFUSERS_IMPORT_ERROR

        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        pipe.scheduler.set_timesteps(pipe.scheduler.config.num_train_timesteps)
        self.pipeline = pipe
        #register all submodules needed, including trainable and non-trainable ones
        self.runtime_modules = {
            "dit": pipe.transformer,
            "vae": pipe.vae,
            "noise_scheduler": pipe.scheduler,
            "text_encoder": pipe.text_encoder,
            "text_encoder_2": pipe.text_encoder_2,
            "text_encoder_3": pipe.text_encoder_3,
            "tokenizer": pipe.tokenizer,
            "tokenizer_2": pipe.tokenizer_2,
            "tokenizer_3": pipe.tokenizer_3,
        }

        #you would need to unfreeze more modules for a real training run
        self.trainable_modules = ["dit"]
        self.primary_train_model = self.runtime_modules["dit"]

        self.runtime_modules["vae"].requires_grad_(False)
        self.runtime_modules["text_encoder"].requires_grad_(False)
        if self.runtime_modules["text_encoder_2"] is not None:
            self.runtime_modules["text_encoder_2"].requires_grad_(False)
        if self.runtime_modules["text_encoder_3"] is not None:
            self.runtime_modules["text_encoder_3"].requires_grad_(False)

        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) \
            if torch.cuda.is_available() else torch.device("cpu")
        self.runtime_modules["vae"].to(device)
        self.runtime_modules["text_encoder"].to(device)
        if self.runtime_modules["text_encoder_2"] is not None:
            self.runtime_modules["text_encoder_2"].to(device)
        if self.runtime_modules["text_encoder_3"] is not None:
            self.runtime_modules["text_encoder_3"].to(device)

        return {
            "model": self.primary_train_model,
            "runtime_modules": self.runtime_modules,
            "trainable_modules": self.trainable_modules,
        }

    def prepare_batch(self, raw_batch: dict[str, Any]) -> BatchState:
        return BatchState(raw_batch=raw_batch)

    def set_train_model(self, model: Any) -> None:
        self.train_model = model
        self.primary_train_model = model
        self.runtime_modules["dit"] = model
        if self.pipeline is not None:
            self.pipeline.transformer = model

    def forward_loss(self, batch_state: BatchState) -> dict[str, Any]:
        if self.train_model is None and self.primary_train_model is None:
            raise RuntimeError("build_modules must be called before forward_loss")
        pixel_values = batch_state.raw_batch["pixel_values"]
        prompts = batch_state.raw_batch["prompts"]
        model = self.train_model if self.train_model is not None else self.primary_train_model

        device = pixel_values.device
        dtype = next(model.parameters()).dtype
        scheduler = self.runtime_modules["noise_scheduler"]
        latents = self._encode_latents(pixel_values, device)
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(prompts, device)
        latents = latents.to(dtype=dtype)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
            dtype=torch.long,
        )
        sigmas = self._sigmas_for_timesteps(timesteps, device=device, dtype=latents.dtype)
        sigmas = sigmas.view(bsz, *([1] * (latents.ndim - 1)))

        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        target = noise - latents

        model_pred = model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        batch_state.loss_dict["loss"] = loss
        batch_state.metrics["loss"] = float(loss.detach().item())
        return {"loss": loss, "metrics": batch_state.metrics, "artifacts": batch_state.artifacts}

    def validation_generate(self, prompts: list[str], output_dir: str) -> dict[str, Any]:
        if self.pipeline is None or self.primary_train_model is None:
            raise RuntimeError("build_modules must be called before validation_generate")
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return {"sample_paths": []}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        transformer = self.pipeline.transformer
        was_training = transformer.training
        transformer.eval()
        maybe_fsdp_model = self.train_model
        full_param_context = nullcontext()
        if maybe_fsdp_model is not None and hasattr(type(maybe_fsdp_model), "summon_full_params"):
            full_param_context = type(maybe_fsdp_model).summon_full_params(
                maybe_fsdp_model, recurse=True
            )

        images: list[str] = []
        with full_param_context, torch.no_grad():
            for index, prompt in enumerate(prompts):
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=20,
                    guidance_scale=4.5,
                ).images[0]
                image_path = output_path / f"sample_{index:03d}.png"
                image.save(image_path)
                images.append(str(image_path))

        if was_training:
            transformer.train()
        return {"sample_paths": images}

    def _encode_latents(self, pixel_values: torch.Tensor, device: torch.device) -> torch.Tensor:
        vae = self.runtime_modules["vae"]
        pixel_values = pixel_values.to(device=device, dtype=vae.dtype)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        return latents

    def _encode_prompts(self, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        encoders = [
            (self.runtime_modules["tokenizer"], self.runtime_modules["text_encoder"]),
            (self.runtime_modules["tokenizer_2"], self.runtime_modules["text_encoder_2"]),
            (self.runtime_modules["tokenizer_3"], self.runtime_modules["text_encoder_3"]),
        ]
        prompt_embeds_list: list[torch.Tensor] = []
        pooled_embeds_list: list[torch.Tensor] = []

        with torch.no_grad():
            for tokenizer, encoder in encoders:
                if tokenizer is None or encoder is None:
                    continue
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(device)
                outputs = encoder(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-2]
                pooled = outputs[0]
                prompt_embeds_list.append(hidden_states)
                pooled_embeds_list.append(pooled)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(pooled_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    def _sigmas_for_timesteps(
        self, timesteps: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        scheduler = self.runtime_modules["noise_scheduler"]
        scheduler_timesteps = scheduler.timesteps.to(device)
        scheduler_sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        step_indices = [(scheduler_timesteps == t).nonzero().item() for t in timesteps]
        return scheduler_sigmas[step_indices]
