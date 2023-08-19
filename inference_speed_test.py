from shape_guided_diffusion import shape_guided_diffusion, init_models, get_segm_image
import torch
import json
import os
import os.path as osp
from torch.profiler.profiler import ProfilerActivity, profile
from timeit import timeit


def run_generation_loop(
    init_image, mask_image, unet, vae, clip, clip_tokenizer, generator, noise
):
    # following the same hyperparams as in `shape_guided_diffusion.ipynb`
    # synchronize to make the results accurate
    torch.cuda.synchronize()
    with torch.autocast("cuda"):
        image = shape_guided_diffusion(
            unet,
            vae,
            clip_tokenizer,
            clip,
            init_image=init_image,
            mask_image=mask_image,
            # Prompt params
            prompt_inversion_inside="truck",
            prompt_inversion_outside="background",
            prompt_inside="lego truck",
            prompt_outside="background",
            num_inside=38,
            num_outside=38,
            # Generation params
            guidance_scale=3.5,
            # Inside-Outside Attention params
            run_cross_attention_mask=True,
            run_self_attention_mask=True,
            self_attn_schedule=1.0,
            cross_attn_schedule=2.5,
            # DDIM Inversion params
            run_inversion=True,
            noise_mixing=0.0,
            # Random seed params
            noise=noise,
            generator=generator,
        )
    torch.cuda.synchronize()


def run_speed_test():
    # run shape-guided diffusion with all parameters the same as `shape_guided_diffusion.ipynb`
    device = torch.device("cuda", 0)
    api_token = os.environ["API_TOKEN"]
    seed = 98374234
    generator = torch.cuda.manual_seed(seed)
    unet, vae, clip, clip_tokenizer = init_models(api_token, device=device)
    noise = torch.randn(
        (1, unet.in_channels, 512 // 8, 512 // 8),
        device=device,
        generator=torch.cuda.manual_seed(seed),
    )

    # prepare input data
    segmentations = json.load(
        open(osp.join(osp.dirname(__file__), "mscoco_shape_prompts", "test.json"))
    )
    image_to_file = {
        image["id"]: image["coco_url"].replace(
            "http://images.cocodataset.org",
            osp.join(osp.dirname(__file__), "assets", "mscoco_shape_prompts"),
        )
        for image in segmentations["images"]
    }
    ann_idx = 838
    mask_image, init_image = get_segm_image(
        segmentations, image_to_file, ann_idx, ann_idx
    )
    mask_image, init_image = [
        _img.resize((512, 512)) for _img in (mask_image, init_image)
    ]

    # record a profiler trace
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            osp.join(osp.dirname(__file__), "logs")
        ),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        run_generation_loop(
            init_image, mask_image, unet, vae, clip, clip_tokenizer, generator, noise
        )
        
    num_runs = 5
    total_time = timeit(
        lambda: run_generation_loop(
            init_image, mask_image, unet, vae, clip, clip_tokenizer, generator, noise
        ),
        number=num_runs,
    )
    print(f"Average Time ({num_runs} runs): {total_time / num_runs:.3f} s")


if __name__ == "__main__":
    run_speed_test()
