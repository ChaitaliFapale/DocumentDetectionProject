import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def mask_to_bbox(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))

    ys, xs = np.where(mask > 0) 
    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w = int(x_max - x_min + 1)
    h = int(y_max - y_min + 1)

    return [int(x_min), int(y_min), w, h]


def build_coco(images_dir, masks_dir, out_json):
    images = []
    annotations = []
    
    ann_id = 1
    img_id = 1

    for file_name in tqdm(sorted(os.listdir(images_dir))):
        if not file_name.lower().endswith(".png"):
            continue

      
        img_path = os.path.join(images_dir, file_name)
        img = Image.open(img_path)
        w, h = img.size

    
        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": w,
            "height": h
        })

       
        mask_path = os.path.join(masks_dir, file_name)

        if not os.path.exists(mask_path):
            print(f"⚠ Mask missing for: {file_name}")
            img_id += 1
            continue

        bbox = mask_to_bbox(mask_path)
        if bbox:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1

        img_id += 1

    
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "passport"}]
    }

    with open(out_json, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"\nCOCO annotations saved to: {out_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out", default="data/annotations/annotations.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    build_coco(args.images, args.masks, args.out)
