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

        import gc
        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            SD3Transformer2DModel,
        )
        from transformers import (
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            T5EncoderModel,
            T5TokenizerFast,
        )

        model_path = args.model_name_or_path

        # Load scheduler — tiny, no issue
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)

        # Load tokenizers — tiny
        tokenizer   = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
        tokenizer_3 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3")

        # Load each encoder one at a time, move to CPU, keep in bfloat16
        # Delete the loader reference immediately so Python can GC the extra copy
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder",
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cpu").eval()
        text_encoder.requires_grad_(False)
        gc.collect()

        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cpu").eval()
        text_encoder_2.requires_grad_(False)
        gc.collect()

        # text_encoder_3 = T5EncoderModel.from_pretrained(
        #     model_path, subfolder="text_encoder_3",
        #     torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        # ).to("cpu").eval()
        # text_encoder_3.requires_grad_(False)
        # gc.collect()
        text_encoder_3 = None
        tokenizer_3 = None

        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae",
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cpu").eval()
        vae.requires_grad_(False)
        gc.collect()

        # Load the DiT last — it's the largest, but now the others
        # have already been loaded and GC'd their loader copies
        dit = SD3Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer",
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
        dit.requires_grad_(True)
        gc.collect()

        self.runtime_modules = {
            "dit":           dit,
            "vae":           vae,
            "noise_scheduler": scheduler,
            "text_encoder":  text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "tokenizer":     tokenizer,
            "tokenizer_2":   tokenizer_2,
            "tokenizer_3":   tokenizer_3,
        }
        self.trainable_modules = ["dit"]
        self.primary_train_model = dit

        # Enable gradient checkpointing to save GPU memory during backward
        if hasattr(dit, "enable_gradient_checkpointing"):
            dit.enable_gradient_checkpointing()

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
        pixel_values = batch_state.raw_batch["pixel_values"]
        prompts = batch_state.raw_batch["prompts"]

        if self.train_model is not None:
            model = self.train_model
        elif self.primary_train_model is not None:
            model = self.primary_train_model
        else:
            raise RuntimeError("build_modules must be called before forward_loss")

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        scheduler = self.runtime_modules["noise_scheduler"]

        latents = self._encode_latents(pixel_values)
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(prompts)

        latents = latents.to(device=device, dtype=dtype)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (bsz,), device=device, dtype=torch.long,
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

    def _encode_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel values to latents on CPU."""
        vae = self.runtime_modules["vae"]
        pixel_values = pixel_values.to(device="cpu", dtype=vae.dtype)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        return latents  # on CPU, caller moves to GPU

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts using all three text encoders on CPU.

        SD3 expects:
          - encoder_hidden_states: (B, seq_len, 4096)
            = T5 hidden states only (4096-dim), since context_embedder projects from 4096.
            CLIP hidden states are concatenated along the sequence dimension first,
            then T5 hidden states are appended — but ALL must have hidden_dim=4096.
            The standard diffusers approach is to zero-pad CLIP to 4096 on the hidden dim,
            then concat all three on seq dim.
          - pooled_projections: (B, 2048)
            = CLIP-L pooled (768) + CLIP-G pooled (1280) concatenated on hidden dim.
            T5 does NOT contribute a pooled embedding.
        """
        encoders = [
            (self.runtime_modules["tokenizer"],   self.runtime_modules["text_encoder"]),    # CLIP-L: hidden 768,  seq 77
            (self.runtime_modules["tokenizer_2"], self.runtime_modules["text_encoder_2"]),  # CLIP-G: hidden 1280, seq 77
            (self.runtime_modules["tokenizer_3"], self.runtime_modules["text_encoder_3"]),  # T5:     hidden 4096, seq 512
        ]

        clip_hidden_states: list[torch.Tensor] = []
        t5_hidden_states: torch.Tensor | None = None
        pooled_embeds_list: list[torch.Tensor] = []

        with torch.no_grad():
            for i, (tokenizer, encoder) in enumerate(encoders):
                if tokenizer is None or encoder is None:
                    continue

                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to("cpu")
                outputs = encoder(input_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[-2]  # (B, seq_len, hidden_dim)

                if i < 2:
                    # CLIP encoders — collect pooled, pad hidden dim to 4096
                    pooled = outputs[0]
                    if pooled.dim() == 3:
                        pooled = pooled[:, 0, :]  # (B, hidden_dim)
                    pooled_embeds_list.append(pooled)

                    # Zero-pad hidden dim from 768/1280 up to 4096
                    target_hidden_dim = 4096
                    pad_size = target_hidden_dim - hidden.shape[-1]
                    if pad_size > 0:
                        pad = torch.zeros(
                            hidden.shape[0], hidden.shape[1], pad_size,
                            dtype=hidden.dtype, device=hidden.device,
                        )
                        hidden = torch.cat([hidden, pad], dim=-1)  # (B, 77, 4096)
                    clip_hidden_states.append(hidden)
                else:
                    # T5 encoder — hidden is already 4096
                    t5_hidden_states = hidden  # (B, 512, 4096)

        # Concatenate along sequence dimension: (B, 77+77+512, 4096) = (B, 666, 4096)
        all_hidden = clip_hidden_states
        if t5_hidden_states is not None:
            all_hidden = all_hidden + [t5_hidden_states]
        prompt_embeds = torch.cat(all_hidden, dim=1)

        # Pooled: CLIP-L + CLIP-G concatenated on hidden dim: (B, 768+1280) = (B, 2048)
        pooled_prompt_embeds = torch.cat(pooled_embeds_list, dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    def _sigmas_for_timesteps(
        self, timesteps: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        scheduler = self.runtime_modules["noise_scheduler"]
        scheduler_timesteps = scheduler.timesteps.cpu().float()
        scheduler_sigmas = scheduler.sigmas.cpu().to(dtype=dtype)

        sigmas = []
        for t in timesteps.cpu():
            # Use closest match rather than exact to avoid float precision issues
            idx = (scheduler_timesteps - float(t)).abs().argmin().item()
            sigmas.append(scheduler_sigmas[idx])

        return torch.stack(sigmas).to(device=device, dtype=dtype)