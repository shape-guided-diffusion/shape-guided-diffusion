import numpy as np
import PIL
from PIL import Image
from pycocotools import mask
import torch

def preprocess_image(image, mult=64, w=512, h=512):
  image = image.resize((w, h), resample=PIL.Image.LANCZOS)
  image = np.array(image).astype(np.float32) 
  # remove alpha channel
  if len(image.shape) == 2:
    image = image[:, :, None]
  else:
    image = image[:, :, (0, 1, 2)]
  # (b, c, w, h)
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image)
  image = image / 255.0
  image = 2. * image - 1.
  return image

def preprocess_segm(segm, num_channels=4, w=512, h=512):
  segm = segm.convert("L")
  segm = segm.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
  segm = np.array(segm).astype(np.float32) / 255.0
  segm = np.tile(segm, (num_channels, 1, 1))
  segm = segm[None]
  segm = 1 - segm  # repaint white, keep black
  segm = torch.from_numpy(segm)
  return segm

def get_segm_image(segmentations, image_to_file, segm_idx, image_idx):
  segm = segmentations["annotations"][segm_idx]["segmentation"]
  image = segmentations["annotations"][image_idx]["image_id"]
  image = image_to_file[image]
  image = np.array(Image.open(image).convert("RGB"))
  h, w, c = image.shape
  rles = mask.frPyObjects(segm, h, w)
  if type(rles) is dict:
      rles = [rles]
  rle = mask.merge(rles)
  segm = mask.decode(rle)
  segm = segm * 255
  
  segm, image = Image.fromarray(segm), Image.fromarray(image)
  return segm, image

def get_text_embeddings(prompt, pipe, return_input_ids=False):
  with torch.autocast("cuda"):
    with torch.no_grad():
      text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
      )
      text_input = text_input.input_ids
      text_input = text_input.to(pipe.device)
      text_embeddings = pipe.text_encoder(text_input)[0]
  if return_input_ids:
    return text_input, text_embeddings
  else:
    return text_embeddings