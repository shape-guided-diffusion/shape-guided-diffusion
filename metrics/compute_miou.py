import argparse
import glob
import json
import numpy as np
import os
from PIL import Image
from pycocotools import mask
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image, convert_PIL_to_numpy
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.colormap import random_color
from detectron2.utils.visualizer import ColorMode, Visualizer
from mask_former.mask_former_model import MaskFormer
from mask_former.config import add_mask_former_config
from mask_former.predictor import BatchedPredictor
from mask_former.data.datasets.register_coco_stuff_10k import COCO_CATEGORIES

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

IMAGE_SIZE = 512

class VisualizerMeta:
    def __init__(self):
        self.stuff_classes = [k["name"] for k in COCO_CATEGORIES]
        self.stuff_colors = [k.get("color", random_color(rgb=True, maximum=1)) for k in COCO_CATEGORIES]

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
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
        "--config-file",
        type=str,
        default="metrics/mask_former/configs/maskformer_R50_bs32_60k.yaml",
    )
    parser.add_argument(
        "--meta-file",
        type=str,
    )
    parser.add_argument(
        "--visualize",
        action="store_true"
    )
    return parser

def setup_cfg(cfg_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg

def get_miou(conf_matrix, num_classes):
    acc = np.full(num_classes, np.nan, dtype=np.float64)
    iou = np.full(num_classes, np.nan, dtype=np.float64)
    tp = conf_matrix.diagonal()[:-1].astype(np.float64)
    pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float64)
    pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float64)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    union = pos_gt + pos_pred - tp
    iou_valid = np.logical_and(acc_valid, union > 0)
    iou[iou_valid] = tp[iou_valid] / union[iou_valid]
    miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
    return iou, miou

def get_img(path):
    image = read_image(path, format="BGR")
    image = Image.fromarray(image)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = convert_PIL_to_numpy(image, None)
    segm_id = int(os.path.basename(path).split(".")[0])
    return image, segm_id

def get_segm(ann, h, w, coco_to_maskformer_category):
    segm = ann["segmentation"]
    category_id = coco_to_maskformer_category[ann["category_id"]]
    rles = mask.frPyObjects(segm, h, w)
    if type(rles) is dict:
        rles = [rles]
    rle = mask.merge(rles)
    segm = mask.decode(rle)
    segm = Image.fromarray(segm)
    segm = segm.resize((IMAGE_SIZE, IMAGE_SIZE))
    segm = np.array(segm)
    segm = torch.from_numpy(segm)
    return segm, category_id

def main():
    args = get_parser().parse_args()
    cfg = setup_cfg(args.config_file)
    maskformer_model = BatchedPredictor(cfg)
    visualizer_meta = VisualizerMeta()

    mscoco_meta = json.load(open(args.meta_file))
    mscoco_anns = {ann["id"]: ann for ann in mscoco_meta["annotations"]}
    mscoco_imgs = {img["id"]: img for img in mscoco_meta["images"]}
    coco_to_maskformer_category = {cat["id"]: i for i, cat in enumerate(COCO_CATEGORIES)}
    num_classes = len(COCO_CATEGORIES)

    mious = []
    img_paths = glob.glob(f"{args.input}/*")
    for i in tqdm(range(0, len(img_paths), args.batch_size)):
        batch_img_paths = img_paths[i:i+args.batch_size]
        batch_imgs, batch_segm_ids, batch_segm_gts, batch_category_ids = [], [], [], []
        for path in batch_img_paths:
            img, segm_id = get_img(path)
            img_id = mscoco_anns[segm_id]["image_id"]
            h, w = mscoco_imgs[img_id]["height"], mscoco_imgs[img_id]["width"]
            segm_gt, category_id = get_segm(mscoco_anns[segm_id], h, w, coco_to_maskformer_category)

            batch_imgs.append(img)
            batch_segm_ids.append(segm_id)
            batch_segm_gts.append(segm_gt)
            batch_category_ids.append(category_id)

        with torch.no_grad():
            batch_segm_preds = maskformer_model((batch_imgs, batch_img_paths))
            batch_segm_preds = torch.stack([segm_pred["sem_seg"] for segm_pred in batch_segm_preds])
            batch_segm_preds = batch_segm_preds.argmax(dim=1)
            batch_segm_preds = batch_segm_preds.detach().cpu().numpy()

        for j, segm_pred in enumerate(batch_segm_preds):
            segm_gt = batch_segm_gts[j]
            # Compute the metric such that the union is always the object mask (segm_gt)
            segm_pred = np.where(segm_gt > 0, segm_pred, num_classes)
            segm_gt = np.where(segm_gt > 0, batch_category_ids[j], num_classes)
            # In the confusion matrix the row = ground truth, column = pred
            # Category (num_classes + 1) corresponds to background
            sample_conf_matrix = np.bincount(
                (num_classes + 1) * segm_pred.reshape(-1) + segm_gt.reshape(-1),
                minlength=((num_classes + 1) * (num_classes + 1)),
            ).reshape((num_classes + 1, num_classes + 1))
            _, miou = get_miou(sample_conf_matrix, num_classes)
            mious.append(miou)

            if args.visualize:
                visualizer = Visualizer(batch_imgs[j][:, :, ::-1], visualizer_meta, instance_mode=ColorMode.IMAGE_BW)
                vis_output = visualizer.draw_sem_seg(
                    torch.from_numpy(segm_pred)
                )
                vis_output = vis_output.get_image()
                vis_output = Image.fromarray(vis_output)
                vis_output = vis_output.save(f"{batch_segm_ids[j]}.png")
     
    print("Total images evaluated:", len(img_paths))
    print("Average sample mIOU:", np.nanmean(mious))

if __name__ == "__main__":
    main()