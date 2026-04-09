# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import xml.etree.ElementTree as ET

xml_path = "/kaggle/input/tbx-11/TBX11K/annotations/xml/tb0329.xml"
tree = ET.parse(xml_path)
root = tree.getroot()

for obj in root.findall("object"):
    name = obj.find("name").text
    bbox = obj.find("bndbox")
    xmin = bbox.find("xmin").text
    ymin = bbox.find("ymin").text
    xmax = bbox.find("xmax").text
    ymax = bbox.find("ymax").text
    print(name, xmin, ymin, xmax, ymax)

import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
# Correct way to read annotation image size
for size in root.findall("size"):
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    print("Annotation expects image size:", width, height)



# Paths
img_path = "/kaggle/input/tbx-11/TBX11K/imgs/tb/tb0329.png"
xml_path = "/kaggle/input/tbx-11/TBX11K/annotations/xml/tb0329.xml"

# Load resized image
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# Parse XML
tree = ET.parse(xml_path)
root = tree.getroot()

# Get original annotation size
size = root.find("size")
orig_w = int(size.find("width").text)
orig_h = int(size.find("height").text)

scale_x = w / orig_w
scale_y = h / orig_h

print("Scale factors:", scale_x, scale_y)

# Draw scaled bounding boxes
for obj in root.findall("object"):
    bbox = obj.find("bndbox")
    xmin = int(int(bbox.find("xmin").text) * scale_x)
    ymin = int(int(bbox.find("ymin").text) * scale_y)
    xmax = int(int(bbox.find("xmax").text) * scale_x)
    ymax = int(int(bbox.find("ymax").text) * scale_y)

    cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)

# Show image
plt.figure(figsize=(6,6))
plt.imshow(img_rgb)
plt.title("TB Bounding Box (Scaled Correctly)")
plt.axis("off")
plt.show()

import os

BASE_PATH = "/kaggle/input/tbx-11/TBX11K"

TRAIN_LIST = os.path.join(BASE_PATH, "lists/TBX11K_train.txt")
VAL_LIST   = os.path.join(BASE_PATH, "lists/TBX11K_val.txt")

with open(TRAIN_LIST) as f:
    train_ids = [line.strip() for line in f.readlines()]

with open(VAL_LIST) as f:
    val_ids = [line.strip() for line in f.readlines()]

print("Train images:", len(train_ids))
print("Val images:", len(val_ids))
print("Sample train IDs:", train_ids[:5])
print("Sample val IDs:", val_ids[:5])

import os
import xml.etree.ElementTree as ET

BASE_PATH = "/kaggle/input/tbx-11/TBX11K"
IMG_DIR = os.path.join(BASE_PATH, "imgs")
XML_DIR = os.path.join(BASE_PATH, "annotations/xml")

def analyze_ids(id_list):
    stats = {
        "total": 0,
        "tb_with_xml": 0,
        "tb_missing_xml": 0,
        "normal": 0,
        "missing_image": 0,
        "multi_lesion": 0
    }

    for rel_path in id_list:
        stats["total"] += 1

        # rel_path example: tb/tb0005.png
        img_path = os.path.join(IMG_DIR, rel_path)

        if not os.path.exists(img_path):
            stats["missing_image"] += 1
            continue

        filename = os.path.basename(rel_path)       # tb0005.png
        img_id = os.path.splitext(filename)[0]      # tb0005

        # TB case
        if rel_path.startswith("tb/"):
            xml_path = os.path.join(XML_DIR, img_id + ".xml")

            if os.path.exists(xml_path):
                stats["tb_with_xml"] += 1

                tree = ET.parse(xml_path)
                root = tree.getroot()
                objs = root.findall("object")
                if len(objs) > 1:
                    stats["multi_lesion"] += 1
            else:
                stats["tb_missing_xml"] += 1

        # Normal case
        elif rel_path.startswith("health/"):
            stats["normal"] += 1

    return stats


print("TRAIN STATS:", analyze_ids(train_ids))
print("VAL STATS:", analyze_ids(val_ids))

import os

YOLO_BASE = "/kaggle/working/tb_yolo"

for split in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_BASE, "images", split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_BASE, "labels", split), exist_ok=True)

print("YOLO directory structure created")

import xml.etree.ElementTree as ET
import cv2

BASE_PATH = "/kaggle/input/tbx-11/TBX11K"
IMG_DIR = os.path.join(BASE_PATH, "imgs")
XML_DIR = os.path.join(BASE_PATH, "annotations/xml")

def convert_xml_to_yolo(rel_img_path, split):
    """
    rel_img_path example: tb/tb0329.png or health/h3703.png
    """
    img_src = os.path.join(IMG_DIR, rel_img_path)
    img_dst = os.path.join(YOLO_BASE, "images", split, os.path.basename(rel_img_path))

    # Copy image
    img = cv2.imread(img_src)
    if img is None:
        return
    cv2.imwrite(img_dst, img)

    label_path = os.path.join(
        YOLO_BASE, "labels", split,
        os.path.splitext(os.path.basename(rel_img_path))[0] + ".txt"
    )

    # Normal image → empty label file
    if rel_img_path.startswith("health/"):
        open(label_path, "w").close()
        return

    # TB image → parse XML
    img_id = os.path.splitext(os.path.basename(rel_img_path))[0]
    xml_path = os.path.join(XML_DIR, img_id + ".xml")

    if not os.path.exists(xml_path):
        open(label_path, "w").close()
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Original annotation size
    size = root.find("size")
    orig_w = int(size.find("width").text)
    orig_h = int(size.find("height").text)

    # Current image size
    h, w, _ = img.shape

    scale_x = w / orig_w
    scale_y = h / orig_h

    with open(label_path, "w") as f:
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")

            xmin = int(int(bbox.find("xmin").text) * scale_x)
            ymin = int(int(bbox.find("ymin").text) * scale_y)
            xmax = int(int(bbox.find("xmax").text) * scale_x)
            ymax = int(int(bbox.find("ymax").text) * scale_y)

            # YOLO format (normalized)
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            f.write(f"0 {x_center} {y_center} {box_w} {box_h}\n")

for rel_path in train_ids:
    convert_xml_to_yolo(rel_path, "train")

for rel_path in val_ids:
    convert_xml_to_yolo(rel_path, "val")

print("Conversion completed")

data_yaml = f"""
path: {YOLO_BASE}
train: images/train
val: images/val

nc: 1
names: ['ActiveTuberculosis']
"""

with open(os.path.join(YOLO_BASE, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("data.yaml created")

import os
import cv2
import matplotlib.pyplot as plt

# List available train images
train_img_dir = os.path.join(YOLO_BASE, "images/train")
train_lbl_dir = os.path.join(YOLO_BASE, "labels/train")

imgs = sorted(os.listdir(train_img_dir))
print("Total train images:", len(imgs))
print("Sample images:", imgs[:5])

# Pick first TB image (starts with 'tb')
sample_img = next(img for img in imgs if img.startswith("tb"))
sample_lbl = sample_img.replace(".png", ".txt")

print("Using sample:", sample_img)

img_path = os.path.join(train_img_dir, sample_img)
lbl_path = os.path.join(train_lbl_dir, sample_lbl)

# Load image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# Draw YOLO boxes
with open(lbl_path) as f:
    for line in f.readlines():
        cls, xc, yc, bw, bh = map(float, line.split())

        xmin = int((xc - bw/2) * w)
        ymin = int((yc - bh/2) * h)
        xmax = int((xc + bw/2) * w)
        ymax = int((yc + bh/2) * h)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title("YOLO Label Sanity Check (Auto-selected)")
plt.axis("off")
plt.show()

!pip install -q ultralytics

from ultralytics import YOLO
print("YOLOv8 ready")

model = YOLO("yolov8s.pt")

model.train(
    data="/kaggle/working/tb_yolo/data.yaml",
    epochs=50,
    imgsz=512,
    batch=16,
    workers=4,
    project="tb_yolo_runs",
    name="yolov8s_tb",
    patience=10,
    device=0
)

from ultralytics import YOLO

MODEL_PATH = "/kaggle/working/runs/detect/tb_yolo_runs/yolov8s_tb/weights/best.pt"
model = YOLO(MODEL_PATH)

print("Model loaded successfully")

import random
import cv2
import matplotlib.pyplot as plt
import os

VAL_IMG_DIR = "/kaggle/working/tb_yolo/images/val"
CONF_TH = 0.25

# Pick random samples
all_imgs = os.listdir(VAL_IMG_DIR)
tb_imgs = [img for img in all_imgs if img.startswith("tb")]
normal_imgs = [img for img in all_imgs if img.startswith("h")]

sample_imgs = random.sample(tb_imgs, 3) + random.sample(normal_imgs, 3)

print("Samples:", sample_imgs)

def visualize_prediction(img_name):
    img_path = os.path.join(VAL_IMG_DIR, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img_path)[0]

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_TH:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"TB {conf:.2f}"

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(
            img_rgb, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    plt.figure(figsize=(5,5))
    plt.imshow(img_rgb)
    plt.title(img_name)
    plt.axis("off")
    plt.show()

for img in sample_imgs:
    visualize_prediction(img)
