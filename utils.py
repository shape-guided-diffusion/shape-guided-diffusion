from difflib import SequenceMatcher
import math
import numpy as np
import PIL
from PIL import Image
import torch

def exponential_schedule(current_over_max, gamma=0.95, multiplier=50):
    return gamma ** (current_over_max * multiplier)

def cosine_schedule(current_over_max):
    return (1 + math.cos(current_over_max * math.pi)) / 2

def slerp(z1, z2, alpha):
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    return (
            torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
            + torch.sin(alpha * theta) / torch.sin(theta) * z2
    )

def init_attention_func(unet,
                        inside_indices, outside_indices, pad_indices,
                        inside_fixed_indices=None, outside_fixed_indices=None,
                        reset=True, self_attn_schedule=None, cross_attn_schedule=None):

    def preprocess_segm(self, segm, batch_size=1, num_channels=4, w=512 // 8, h=512 // 8, repaint=True):
        segm = segm.convert("L")
        segm = segm.resize((w, h), resample=PIL.Image.NEAREST)
        segm = np.array(segm).astype(np.float32) / 255.0
        segm = np.tile(segm, (batch_size, num_channels, 1, 1))
        if repaint:
            segm = 1 - segm  # repaint white, keep black
        segm = torch.from_numpy(segm)
        return segm

    def new_attention(self, query, key, value, sequence_length=None, dim=None):
        batch_size_attention = query.shape[0]

        if self.mask_image is not None:
            image_dim = int(math.sqrt(query.shape[1]))
            segm = self.preprocess_segm(
                self.mask_image,
                batch_size=batch_size_attention,
                num_channels=1,
                w=image_dim,
                h=image_dim
            ).to(query.device)
            segm = segm.view(batch_size_attention, 1, -1)
            segm = segm.permute((0, 2, 1))

        if dim is None:
            dim = query.shape[2] * self.heads

        if sequence_length is None:
            sequence_length = query.shape[1]

        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )

        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                    torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
            )
            attn_slice = attn_slice.softmax(dim=-1)

            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    attn_slice = attn_slice * (
                                1 - self.last_attn_slice_mask) + self.last_attn_slice * self.last_attn_slice_mask
                else:
                    attn_slice = self.last_attn_slice
                self.use_last_attn_slice = False

            if self.save_last_attn_slice:
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False

            if not self.mask_tokens_attn and self.is_token_attn:
                attn_slice *= cross_attn_schedule
                if inside_fixed_indices is not None:
                    attn_slice[:, :, inside_fixed_indices] /= cross_attn_schedule
                if outside_fixed_indices is not None:
                    attn_slice[:, :, outside_fixed_indices] /= cross_attn_schedule

            if self.mask_image is not None and self.mask_tokens_attn:

                inside_slice = attn_slice[:, :, inside_indices] * (1 - segm)
                outside_slice = attn_slice[:, :, outside_indices] * segm
                if cross_attn_schedule is not None and type(cross_attn_schedule) is not str:
                    inside_scale = cross_attn_schedule
                    outside_scale = cross_attn_schedule
                else:
                    inside_scale = 1.0
                    outside_scale = 1.0
                attn_slice[:, :, inside_indices] = inside_slice * inside_scale
                attn_slice[:, :, outside_indices] = outside_slice * outside_scale

                if inside_fixed_indices is not None:
                    attn_slice[:, :, inside_fixed_indices] /= inside_scale

                if outside_fixed_indices is not None:
                    attn_slice[:, :, outside_fixed_indices] /= outside_scale

                attn_slice[:, :, pad_indices] = 0
                self.mask_tokens_attn = False

            if self.mask_image is not None and self.mask_self_attn:
                _, mask_inside_indices, _ = (1 - segm).nonzero(as_tuple=True)
                _, mask_outside_indices, _ = segm.nonzero(as_tuple=True)
                insideout_mask = torch.ones_like(attn_slice) * (1 - segm)
                insideout_mask[:, :, mask_outside_indices] = segm

                if self_attn_schedule is not None and type(self_attn_schedule) is not str:
                    self_attn_scale = self_attn_schedule
                elif self_attn_schedule == "linear":
                    self_attn_scale = 1 - self.timestep
                elif self_attn_schedule == "exponential":
                    self_attn_scale = exponential_schedule(self.timestep)
                elif self_attn_schedule == "cosine":
                    self_attn_scale = cosine_schedule(self.timestep)
                else:
                    raise NotImplementedError

                attn_slice = attn_slice * ((insideout_mask - 1) * self_attn_scale + 1)
                self.mask_self_attn = False

            if self.save_attn_name:
                temp_attn_name_slice = self.attn_name_slice.get(self.save_attn_name)
                if temp_attn_name_slice is None:
                    temp_attn_name_slice = attn_slice
                else:
                    temp_attn_name_slice += attn_slice
                self.attn_name_slice[self.save_attn_name] = temp_attn_name_slice
                self.save_attn_name = None

            attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module._attention = new_attention.__get__(module, type(module))
            module.preprocess_segm = preprocess_segm.__get__(module, type(module))
            if "attn1" in name:
                module.is_self_attn = True
                module.is_token_attn = False
            elif "attn2" in name:
                module.is_self_attn = False
                module.is_token_attn = True
            if reset:
                module.timestep = None
                module.last_attn_slice = None
                module.use_last_attn_slice = False
                module.save_last_attn_slice = False
                module.attn_name_slice = {}
                module.save_attn_name = None
                module.mask_image = None
                module.mask_tokens_attn = False
                module.mask_self_attn = False
                module.self_attn_scale = 1
                module.self_attn_schedule = None

def save_attention(unet, save_attn_name=None):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.save_attn_name = save_attn_name

def save_mask_image(unet, mask=None):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.mask_image = mask

def use_mask_tokens_attention(unet, mask=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.mask_tokens_attn = mask

def use_mask_self_attention(unet, mask=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.mask_self_attn = mask

def set_timestep(unet, timestep=None):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.timestep = timestep


def get_tokens_embedding(clip_tokenizer, clip, device, prompt):
    tokens = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True
    )
    input_ids = tokens.input_ids.to(device)
    embedding = clip(input_ids).last_hidden_state
    return tokens, embedding

def display_prompt_tokens(clip_tokenizer, prompt):
    tokens = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True
    ).input_ids[0]
    for idx, token in enumerate(tokens):
        decoded_token = clip_tokenizer.decode(token)
        if decoded_token == "<|startoftext|>":
            continue
        elif decoded_token == "<|endoftext|>":
            break
        else:
            print(idx, "->", decoded_token)

def inversion(x, model, scheduler, **kwargs):
    seq = scheduler.timesteps
    seq = torch.flip(seq, dims=(0,))
    b = scheduler.betas
    b = b.to(x.device)
    
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        if kwargs.get("run_inversion", False):
            print("running inversion...")
            seq_iter = seq_next[1:]
            seq_next_iter = seq[1:]
        else:
            seq_iter = reversed(seq)
            seq_next_iter = reversed(seq_next)
            
        x0_preds = []
        xs = [x]
        for index, (i, j) in enumerate(zip(seq_iter, seq_next_iter)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = (1 - b).cumprod(dim=0).index_select(0, t.long())
            if next_t.sum() == -next_t.shape[0]:
                at_next = torch.ones_like(at)
            else:
                at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
            
            xt = xs[-1].to(x.device)

            # Since this is inversion, the timestep is reversed
            # set_timestep(model, (len(seq_iter) - index) / len(seq_iter))
            # Set the timestep to 0.0 to enforce hard attention across all timesteps
            set_timestep(model, 0.0)
            use_mask_tokens_attention(model, kwargs["run_cross_attention_mask"])
            use_mask_self_attention(model, kwargs["run_self_attention_mask"])

            if "conditional" in kwargs:
                c = kwargs["conditional"]
                et = model(xt, t, encoder_hidden_states=c).sample
            else:
                et = model(xt, t).sample
                
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            eta = kwargs.get("eta", 0)
            if eta == 0:
                c1 = 0
            else:
                c1 = (
                    eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def postprocess(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)

def apply_mask(image, mask_image):
    image = np.array(image)
    mask_image = np.array(mask_image)[:, :, None] / 255
    return Image.fromarray((image * mask_image).astype(np.uint8))

def noise_to_latent(latent, noise_pred, at, eta, at_next):
    x0_t = (latent - noise_pred * (1 - at).sqrt()) / at.sqrt()
    if eta == 0:
        c1 = 0
    else:
        c1 = (
            eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    latent = at_next.sqrt() * x0_t + c1 * torch.randn_like(latent) + c2 * noise_pred
    return latent


def latent_to_image(vae, latent):
    latent = latent / 0.18215
    image = vae.decode(latent.to(vae.dtype)).sample
    image = postprocess(image)
    return image

def compute_fixed_indices(tokens_inversion, tokens, num_tokens=38):
    first_pad = tokens["attention_mask"].sum().item() - 1
    tokens_inversion = tokens_inversion.input_ids.numpy()[0]
    tokens = tokens.input_ids.numpy()[0]
    fixed_indices = []
    for name, a0, a1, b0, b1 in SequenceMatcher(
        None, tokens_inversion, tokens
    ).get_opcodes():
        if name == "equal" and b0 < first_pad:
            b1 = min(b1, num_tokens)
            fixed_indices += list(range(b0, b1))
    return torch.tensor(fixed_indices)[1:]