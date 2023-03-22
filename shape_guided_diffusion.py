from diffusers import (
    AutoencoderKL, UNet2DConditionModel, DDIMScheduler
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor

import torch
from tqdm import tqdm
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
        init_image=None,
        mask_image=None,
        invert_mask=False,
        # Prompt params
        prompt_inside=None,
        prompt_outside=None,
        prompt_inversion_inside=None,
        prompt_inversion_outside=None,
        num_inside=38,
        num_outside=38,
        # Generation params
        guidance_scale=3.5,
        steps=50,
        copy_background=True,
        # Inside-Outside Attention params
        run_cross_attention_mask=True,
        run_self_attention_mask=True,
        self_attn_schedule=1.0,
        cross_attn_schedule=2.5,
        # DDIM Inversion params
        run_inversion=True,
        eta=0.0,
        noise_mixing=0.0,
        # Random seed params
        generator=None,
        noise=None,
        # Misc params
        width=512,
        height=512,
        init_image_strength=1.0,
        save_attention_names=[]
):
    device = unet.device
    init_image = preprocess_image(init_image)
    init_image = init_image.to(device)
    mask = preprocess_segm(mask_image)
    mask = mask.to(device)

    if invert_mask:
        mask = 1.0 - mask

    prompt_inside_indices = torch.tensor([i for i in range(1, num_inside + 1)])
    prompt_outside_indices = torch.tensor([i for i in range(num_inside + 1, num_inside + num_outside + 1)])
    prompt_pad_indices = torch.tensor([0] + [i for i in range(num_inside + num_outside + 1, 77)])

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

        if copy_background:
            if run_inversion:
                init_latent_mask = xts[-t_index - 1].to(device)
            else:
                init_latent_mask = scheduler.add_noise(init_latent, noise, seq_iter[i])
            prompt_latent = (init_latent_mask * mask) + (prompt_latent * (1 - mask))

    save_mask_image(unet, mask=None)
    prompt_image = latent_to_image(vae, prompt_latent)

    return prompt_image

def init_models(
    huggingface_access_token,
    torch_dtype = torch.float16,
    device = "cuda",
    model_precision_type = "fp16",
    model_path_clip = "openai/clip-vit-large-patch14",
    model_path_diffusion = "runwayml/stable-diffusion-v1-5"
):
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
    clip_model = CLIPModel.from_pretrained(
        model_path_clip,
        torch_dtype=torch_dtype
    )
    clip = clip_model.text_model
    unet = UNet2DConditionModel.from_pretrained(
        model_path_diffusion,
        subfolder="unet",
        use_auth_token=huggingface_access_token,
        revision=model_precision_type,
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        model_path_diffusion,
        subfolder="vae",
        use_auth_token=huggingface_access_token,
        revision=model_precision_type,
        torch_dtype=torch.float16
    )
    unet.to(device)
    vae.to(device)
    clip.to(device)
    return unet, vae, clip, clip_tokenizer

def init_safety_checker(
    device = "cuda",
    model_path_clip = "openai/clip-vit-large-patch14",
    model_path_safety="CompVis/stable-diffusion-safety-checker"
):
    feature_extractor = CLIPFeatureExtractor.from_pretrained(model_path_clip)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_path_safety)
    safety_checker.to(device)
    return feature_extractor, safety_checker

def check_image(feature_extractor, safety_checker, image):
    safety_checker_input = feature_extractor(image, return_tensors="pt")
    safety_checker_input = safety_checker_input.to(safety_checker.device)
    image, has_nsfw_concept = safety_checker(
        images=image, 
        clip_input=safety_checker_input.pixel_values
    )
    return image, has_nsfw_concept[0]