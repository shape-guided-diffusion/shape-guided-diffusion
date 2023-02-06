import argparse
from PIL import Image
import json
import multiprocessing
import os
import uuid

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
    )
    parser.add_argument(
        "--meta-file",
        type=str,
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=10
    )
    return parser

def crop_image(ann, width, height, img):
    x, y, w, h = ann["bbox"]
    # Resize the bbox to img.width, img.height
    x_scale, y_scale = img.width / width, img.height / height
    x1, x2, y1, y2 = x, x + w, y, y + h
    x1, x2 = x1 * x_scale, x2 * x_scale
    y1, y2 = y1 * y_scale, y2 * y_scale
    return img.crop((x1, y1, x2, y2))

def process_image(ann, img_meta, input, temp_input):
    width, height = img_meta["width"], img_meta["height"]
    img = Image.open(f"{input}/{ann['id']}.png")
    img = crop_image(ann, width, height, img)
    img.save(f"{temp_input}/{ann['id']}.png")

def main():
    args = get_parser().parse_args()
    temp_input = str(uuid.uuid4())
    os.mkdir(temp_input)
    mscoco_meta = json.load(open(args.meta_file))

    mscoco_anns = mscoco_meta["annotations"]
    mscoco_imgs = {img["id"]: img for img in mscoco_meta["images"]}
    mscoco_imgs = [mscoco_imgs[ann["image_id"]] for ann in mscoco_anns]

    process_image(mscoco_anns[0], mscoco_imgs[0], args.input, temp_input)

    with multiprocessing.Pool(processes=args.nthreads) as pool:
        pool.starmap(process_image, [(ann, img_meta, args.input, temp_input) for ann, img_meta in zip(mscoco_anns, mscoco_imgs)])

    # Capture the temp folder in stdout
    print(temp_input)
    
if __name__ == "__main__":
    main()