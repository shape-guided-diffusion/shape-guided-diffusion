import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import math
import PIL

import torch

from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, DDIMScheduler
)
from difflib import SequenceMatcher

import sys
from preprocess import preprocess_image, preprocess_segm, get_segm_image
from utils import (
    save_mask_image,
    save_attention,
    get_tokens_embedding,
    inversion,
    noise_to_latent,
    latent_to_image,
    use_mask_tokens_attention,
    use_mask_self_attention,
    init_attention_func,
    set_timestep,
    slerp,
    compute_fixed_indices
)


@torch.no_grad()
def shape_guided_diffusion(
        unet,
        vae,
        clip_tokenizer,
        clip,
        prompt_inside_indices,
        prompt_outside_indices,
        prompt_pad_indices,
        prompt_inside=None,
        prompt_outside=None,
        prompt_inversion_inside=None,
        prompt_inversion_outside=None,
        num_inside=38,
        num_outside=38,
        guidance_scale=7.5,
        steps=50,
        generator=None,
        width=512,
        height=512,
        init_image=None,
        init_image_strength=1.0,
        eta=0.0,
        mask=None,
        mask_image=None,
        run_inversion=False,
        run_cross_attention_mask=False,
        run_self_attention_mask=False,
        save_attention_names=[],
        noise=None,
        noise_mixing=0.0,
        self_attn_schedule=None,
        cross_attn_schedule=None,
):
    device = unet.device
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000
    )
    scheduler.set_timesteps(steps)
    init_latent = vae.encode(
        init_image
    ).latent_dist.sample(generator=generator) * 0.18215
    t_start = steps - int(steps * init_image_strength)

    _, embedding_unconditional = get_tokens_embedding(clip_tokenizer, clip, device, "")
    prompt_inside_collect = torch.tensor([i for i in range(1, num_inside + 1)])
    prompt_outside_collect = torch.tensor([i for i in range(1, num_outside + 1)])

    embedding_inversion = embedding_unconditional.clone()
    tokens_inversion_inside, embedding_inversion_inside = get_tokens_embedding(clip_tokenizer, clip, device, prompt_inversion_inside)
    tokens_inversion_outside, embedding_inversion_outside = get_tokens_embedding(clip_tokenizer, clip, device, prompt_inversion_outside)
    embedding_inversion[:, prompt_inside_indices, :] = embedding_inversion_inside[:, prompt_inside_collect, :]
    embedding_inversion[:, prompt_outside_indices, :] = embedding_inversion_outside[:, prompt_outside_collect, :]

    embedding_conditional = embedding_unconditional.clone()
    tokens_inside, embedding_inside = get_tokens_embedding(clip_tokenizer, clip, device, prompt_inside)
    tokens_outside, embedding_outside = get_tokens_embedding(clip_tokenizer, clip, device, prompt_outside)
    embedding_conditional[:, prompt_inside_indices, :] = embedding_inside[:, prompt_inside_collect, :]
    embedding_conditional[:, prompt_outside_indices, :] = embedding_outside[:, prompt_outside_collect, :]

    fixed_indices_inside = compute_fixed_indices(tokens_inversion_inside, tokens_inside, num_inside)
    fixed_indices_outside = compute_fixed_indices(tokens_inversion_outside, tokens_outside, num_outside)
    fixed_indices_outside = prompt_outside_indices[fixed_indices_outside - 1]

    init_attention_func(
        unet,
        prompt_inside_indices,
        prompt_outside_indices,
        prompt_pad_indices,
        self_attn_schedule=self_attn_schedule,
        cross_attn_schedule=cross_attn_schedule,
    )

    save_mask_image(unet, mask=mask_image)

    if noise is None:
        noise = torch.randn(
            (1, unet.in_channels, height // 8, width // 8),
            device=device,
            generator=generator
        )

    if run_inversion:
        assert init_image is not None, f"init_image should not be None if running inversion"
        if "inversion" in save_attention_names:
            save_attention(unet, save_attn_name="inversion")
        xts, _ = inversion(
            init_latent,
            unet,
            scheduler,
            conditional=embedding_inversion,
            run_inversion=True,
            eta=0.0,
            run_cross_attention_mask=run_cross_attention_mask,
            run_self_attention_mask=run_self_attention_mask
        )
        init_latent = xts[-t_start - 1].to(device)
        start_latent = init_latent
    else:
        start_latent = scheduler.add_noise(init_latent, noise, scheduler.timesteps[t_start])

    init_attention_func(
        unet,
        prompt_inside_indices,
        prompt_outside_indices,
        prompt_pad_indices,
        inside_fixed_indices=fixed_indices_inside,
        outside_fixed_indices=fixed_indices_outside,
        self_attn_schedule=self_attn_schedule,
        cross_attn_schedule=cross_attn_schedule,
        reset=False
    )

    timesteps = scheduler.timesteps[t_start:]
    seq = torch.flip(timesteps, dims=(0,))
    seq_next = [-1] + list(seq[:-1])
    b = scheduler.betas
    b = b.to(device)
    seq_iter = reversed(seq)
    seq_next_iter = reversed(seq_next)
    n = init_latent.size(0)

    prompt_latent = start_latent
    if mask is not None:
        prompt_latent_slerped = slerp(prompt_latent, noise, noise_mixing)
        prompt_latent = prompt_latent * mask + prompt_latent_slerped * (1 - mask)

    for i, (t, next_t) in tqdm(enumerate(zip(seq_iter, seq_next_iter)), total=len(seq_iter)):
        t_index = t_start + i
        t = (torch.ones(n) * t).to(device)
        next_t = (torch.ones(n) * next_t).to(device)
        at = (1 - b).cumprod(dim=0).index_select(0, t.long())
        if next_t.sum() == -next_t.shape[0]:
            at_next = torch.ones_like(at)
        else:
            at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())

        set_timestep(unet, i / len(seq_iter))

        # prompt_noise_pred_cond
        if "prompt_noise_pred_cond" in save_attention_names:
            save_attention(unet, save_attn_name="prompt_noise_pred_cond")
        use_mask_tokens_attention(unet, run_cross_attention_mask)
        use_mask_self_attention(unet, run_self_attention_mask)
        prompt_noise_pred_cond = unet(prompt_latent, t, encoder_hidden_states=embedding_conditional).sample

        # prompt_noise_pred_uncond
        if "prompt_noise_pred_uncond" in save_attention_names:
            save_attention(unet, save_attn_name="prompt_noise_pred_uncond")
        use_mask_tokens_attention(unet, run_cross_attention_mask)
        use_mask_self_attention(unet, run_self_attention_mask)
        prompt_noise_pred_uncond = unet(prompt_latent, t, encoder_hidden_states=embedding_unconditional).sample

        if isinstance(guidance_scale, list):
            cf_gd_scale = guidance_scale[t_index]
        else:
            cf_gd_scale = guidance_scale

        if cf_gd_scale == -1:
            prompt_noise_pred = prompt_noise_pred_cond
        else:
            prompt_noise_pred = prompt_noise_pred_cond + cf_gd_scale * (
                        prompt_noise_pred_cond - prompt_noise_pred_uncond)

        prompt_latent = noise_to_latent(prompt_latent, prompt_noise_pred, at, eta, at_next)

        if mask is not None:
            if run_inversion:
                init_latent_mask = xts[-t_index - 1].to(device)
            else:
                init_latent_mask = scheduler.add_noise(init_latent, noise, seq_iter[i])
            prompt_latent = (init_latent_mask * mask) + (prompt_latent * (1 - mask))

    save_mask_image(unet, mask=None)
    prompt_image = latent_to_image(vae, prompt_latent)

    return prompt_image
