import argparse
import clip
import glob
import json
import os
import numpy as np
import PIL
from PIL import Image
import torch
from tqdm import tqdm

IMAGE_SIZE = 512

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="A directory with images where the basename of each file corresponds to an MS-COCO segmentation id",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--meta-file",
        type=str,
    )
    return parser

def get_embeddings(model, samples, mode):
    with torch.no_grad():
        if mode == "image":
            embeddings = model.encode_image(samples)
        elif mode == "text":
            embeddings = model.encode_text(samples)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
    return embeddings

def main():
    args = get_parser().parse_args()
    device = "cuda"
    model, preprocess = clip.load("ViT-L/14")
    model = model.to(device)
    model = model.eval()

    mscoco_meta = json.load(open(args.meta_file))
    mscoco_anns = {ann["id"]: ann for ann in mscoco_meta["annotations"]}

    clip_scores = []
    img_paths = glob.glob(f"{args.input}/*")
    for i in tqdm(range(0, len(img_paths), args.batch_size)):
        batch_img_paths = img_paths[i:i+args.batch_size]
        batch_imgs = [Image.open(image) for image in batch_img_paths]
        batch_imgs = [img.resize((IMAGE_SIZE, IMAGE_SIZE)) for img in batch_imgs]

        batch_imgs = [preprocess(image)[None, :, :, :] for image in batch_imgs]
        batch_imgs = torch.vstack(batch_imgs)
        batch_imgs = batch_imgs.to(device)
        batch_imgs = get_embeddings(model, batch_imgs, "image")

        batch_segm_ids = [int(os.path.basename(path).split(".")[0]) for path in batch_img_paths]
        batch_texts = [mscoco_anns[segm_id]["text"] for segm_id in batch_segm_ids]
        batch_texts = clip.tokenize(batch_texts)
        batch_texts = batch_texts.to(device)
        batch_texts = get_embeddings(model, batch_texts, "text")

        sims = batch_imgs @ batch_texts.T
        for j in range(len(sims)):
            clip_scores.append(sims[j, j].item())

    print("Total images evaluated:", len(img_paths))
    print("CLIP Score:", np.mean(clip_scores))

if __name__ == "__main__":
    main()