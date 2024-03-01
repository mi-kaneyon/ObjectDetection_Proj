
from PIL import Image
import json
import os
import numpy as np
from datetime import datetime

# パラメータ設定
image_resolution = (640, 480)
background_colors = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0)}
defect_types = ["deaddot", "brightdot", "discoloration"]
output_dir = "lcd_images"
os.makedirs(output_dir, exist_ok=True)

# COCO JSONファイルの初期化
coco_json = {
    "info": {
        "year": datetime.now().year,
        "version": "1.0",
        "description": "LCD Image Dataset",
        "contributor": "",
        "url": "",
        "date_created": datetime.now().isoformat()
    },
    "images": [],
    "annotations": [],
    "licenses": [],
    "categories": [
        {"id": 0, "name": "ok"},
        {"id": 1, "name": "deaddot"},
        {"id": 2, "name": "brightdot"},
        {"id": 3, "name": "discoloration"}
    ]
}

def create_defect(image, defect_type):
    x, y = np.random.randint(0, image.width), np.random.randint(0, image.height)
    if defect_type == "deaddot":
        if image.getpixel((x, y)) == background_colors["white"]:
            image.putpixel((x, y), (0, 0, 0))
    elif defect_type == "brightdot":
        if image.getpixel((x, y)) != background_colors["white"]:
            image.putpixel((x, y), (255, 255, 255))
    elif defect_type == "discoloration":
        current_color = image.getpixel((x, y))
        new_color = tuple(np.random.randint(0, 256, size=3))
        if current_color != new_color:
            image.putpixel((x, y), new_color)
    return x, y

def create_image_and_annotation(color_name, color, img_id, defect_type=None):
    # 不良画像の場合、ファイル名に不良タイプを含める
    prefix = defect_type if defect_type else color_name
    file_name = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
    image_path = os.path.join(output_dir, file_name)
    
    if not defect_type:
        # 良品画像の生成
        image = Image.new("RGB", image_resolution, color)
        image.save(image_path)
    else:
        # 不良画像の生成
        image = Image.new("RGB", image_resolution, color)
        x, y = create_defect(image, defect_type)
        image.save(image_path)
        annotation = {
            "id": len(coco_json["annotations"]) + 1,
            "image_id": img_id,
            "category_id": defect_types.index(defect_type) + 1,
            "segmentation": [],
            "area": 1,
            "bbox": [x, y, 1, 1],
            "iscrowd": 0
        }
        coco_json["annotations"].append(annotation)

    coco_json["images"].append({
        "id": img_id,
        "width": image_resolution[0],
        "height": image_resolution[1],
        "file_name": file_name,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": datetime.now().isoformat()
    })

# 各背景色ごとに全てのピクセル位置で不良を生成
img_id = 1
for color_name, color in background_colors.items():
    for y in range(image_resolution[1]):
        for x in range(image_resolution[0]):
            # 良品画像の生成
            create_image_and_annotation(color_name, color, img_id)
            img_id += 1

            # 不良画像の生成
            for defect_type in defect_types:
                create_image_and_annotation(color_name, color, img_id, defect_type)
                img_id += 1

# JSONファイルの保存
json_path = os.path.join(output_dir, "coco_corrected.json")
with open(json_path, "w") as json_file:
    json.dump(coco_json, json_file, indent=4)
