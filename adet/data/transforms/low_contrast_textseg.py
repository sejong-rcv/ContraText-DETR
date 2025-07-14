import os
import random
import numpy as np
import re
import json
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage import sobel
from detectron2.structures import BoxMode

CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',
            '0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@',
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
            'R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b',
            'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',
            't','u','v','w','x','y','z','{','|','}','~']

def calculate_contrast(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return np.std(gray)

def get_crop_contrast(image, bbox):
    x_min, y_min, width, height = bbox
    crop = image.crop((x_min, y_min, x_min + width, y_min + height))
    return calculate_contrast(crop)

def interpolate_points(points, num_points):
    points = np.array(points).reshape(-1, 2)
    num_existing_points = len(points)
    new_points = []
    for i in range(num_existing_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_existing_points]
        new_points.append(p1)
        for j in range(1, num_points // num_existing_points):
            new_points.append(p1 + j * (p2 - p1) / (num_points // num_existing_points))
    return np.array(new_points).flatten().tolist()

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]


def decode_text_to_rec(text, max_length=25):
    rec = [CTLABELS.index(char) if char in CTLABELS else 96 for char in text]
    rec += [96] * (max_length - len(rec))
    return rec[:max_length]

def adjust_colors(crop, avg_color, mask):
    if crop.mode != 'RGB':
        crop = crop.convert('RGB')
    crop_array = np.array(crop)

    mask_3d = np.stack([mask]*3, axis=-1)
    random_contrast_value = random.uniform(35, 50)
    add_or_subtract = random.choice([True, False])

    if add_or_subtract:
        avg_color_adjusted = avg_color + random_contrast_value
    else:
        avg_color_adjusted = avg_color - random_contrast_value
    
    if avg_color_adjusted.mean() < 10:
        avg_color_adjusted = avg_color + random_contrast_value * 2
    
    final_color_adjustment = avg_color_adjusted
    adjusted_array = np.where(mask_3d, final_color_adjustment, crop_array)
    return Image.fromarray(np.uint8(adjusted_array))

def get_avg_color(image, pos, size, margin=0):
    x, y = pos
    width, height = size
    left = max(x - margin, 0)
    top = max(y - margin, 0)
    right = min(x + width + margin, image.width)
    bottom = min(y + height + margin, image.height)
    region = image.crop((left, top, right, bottom))
    region_array = np.array(region)
    return np.mean(region_array, axis=(0, 1))

def pick_ts_ann(cfg):
    ts_seg_imgs = os.listdir(cfg.DATASETS.TEXTSEG_IMG_PATH)
    ts_img_name = random.choice(ts_seg_imgs) 

    ts_img_path = os.path.join(cfg.DATASETS.TEXTSEG_IMG_PATH, ts_img_name)
    ts_ann_path = os.path.join(cfg.DATASETS.TEXTSEG_ANN_PATH, f'{ts_img_name[:-11]}_anno.json')

    base_image = Image.open(ts_img_path)
    base_width, base_height = base_image.size

    with open(ts_ann_path, 'r') as file:
        json_data = json.load(file)

    horizontal_tt_ann = []
    for key, ann_data in json_data.items():
        text = ann_data['text']
        bbox = ann_data['bbox']
        x_coords = bbox[::2]
        y_coords = bbox[1::2]

        box_width = max(x_coords) - min(x_coords)
        box_height = max(y_coords) - min(y_coords)

        if box_height > 52 and box_width > 101:
            horizontal_tt_ann.append({
                "text": text,
                "bbox": bbox
            })

    return ts_img_path, horizontal_tt_ann

def jaccard(box_a, box_b):
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union != 0 else 0

def find_non_overlapping_position(crop_width, crop_height, image_width, image_height, existing_boxes, iou_threshold=0.05):
    trials = 50
    for _ in range(trials):
        random_x = random.randint(0, image_width - crop_width)
        random_y = random.randint(0, image_height - crop_height)
        new_box = (random_x, random_y, random_x + crop_width, random_y + crop_height)
        overlaps = [jaccard(new_box, box) for box in existing_boxes]

        if all(overlap < iou_threshold for overlap in overlaps):
            return random_x, random_y
    return None

def load_bbox_distribution(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            bbox_sizes = json.load(json_file)
        return bbox_sizes
    else:
        raise FileNotFoundError(f"{file_path} not found!")

def calculate_average_box_size(boxes):
    total_width = 0
    total_height = 0
    
    for box in boxes:
        x_min, y_min, x_max, y_max = box 
        width = x_max - x_min 
        height = y_max - y_min  

        total_width += width
        total_height += height
    
    average_width = total_width / len(boxes)
    average_height = total_height / len(boxes)
    
    return average_width, average_height

def synthesize_textseg(image, boxes, cfg):
    bbox_distribution_file_path = cfg.DATASETS.BBOX_DISTRIBUTION_PATH
    mpsc_bbox_scale_distribution = load_bbox_distribution(bbox_distribution_file_path)

    while True:
        ts_img_path, ts_ann = pick_ts_ann(cfg)
        if len(ts_ann) != 0:
            break

    if len(ts_ann) > 5:
        synth_num = random.randint(1, 5)
    else:
        synth_num = random.randint(1, len(ts_ann))

    selected_anns = random.sample(ts_ann, synth_num)

    base_image = Image.open(ts_img_path)
    target_image = Image.fromarray(image)
    target_width, target_height = target_image.size
    base_width, base_height = base_image.size

    crops, texts, new_annotations = [], [], []

    for ann in selected_anns:
        text = ann['text']
        bbox = ann['bbox']

        x_coords = bbox[::2]
        y_coords = bbox[1::2]

        left = min(x_coords)
        top = min(y_coords)
        right = max(x_coords)
        bottom = max(y_coords)

        crop_img = base_image.crop((left, top, right, bottom))
        crop_width, crop_height = crop_img.size
        crop_aspect_ratio = crop_width / crop_height
        target_aspect_ratio = target_width / target_height
        
        average_scale_factor = np.mean(mpsc_bbox_scale_distribution)
        variation = random.uniform(-0.005, 0.001)
        scale_factor = max(0.0001, average_scale_factor + variation)
        target_area = target_width * target_height * scale_factor

        aspect_ratio = crop_width / crop_height
        new_height = int(np.sqrt(target_area / aspect_ratio))
        new_width = int(new_height * aspect_ratio)

        if (new_width * 4 < crop_width) or (new_height * 4 < crop_height):
            continue

        crop_img = crop_img.resize((new_width, new_height), Image.ANTIALIAS)
        crops.append(crop_img)
        texts.append(text)

    existing_boxes = [tuple(box) for box in boxes]

    for crop, text in zip(crops, texts):
        position = find_non_overlapping_position(crop.width, crop.height, target_image.width, target_image.height, existing_boxes)
        if position is not None:
            random_x, random_y = position
        else:
            continue

        mask = np.array(crop)
        mask = (mask != 0) & (mask < 200)
        mask_pil = Image.fromarray(np.uint8(mask) * 255)

        avg_color = get_avg_color(target_image, (random_x, random_y), crop.size)
        adjusted_crop = adjust_colors(crop, avg_color, mask)
        
        temp_image = target_image.copy()
        temp_image.paste(adjusted_crop, (random_x, random_y), mask_pil)
        
        new_bbox = [random_x, random_y, crop.width, crop.height]
        if get_crop_contrast(temp_image, new_bbox) >= 30:
            continue
            
        target_image.paste(adjusted_crop, (random_x, random_y), mask_pil)

        new_box_mode = BoxMode.XYWH_ABS
        new_polygon = interpolate_points([random_x, random_y, random_x + crop.width, random_y, random_x + crop.width, random_y + crop.height, random_x, random_y + crop.height], 16)
        rec = decode_text_to_rec(text)

        existing_boxes.append((random_x, random_y, random_x + crop.width, random_y + crop.height))

        new_annotation = {
            'iscrowd': 0,
            'bbox': new_bbox,
            'rec': rec,
            'category_id': 0,
            'polygons': new_polygon,
            'text': rec,
            'bbox_mode': new_box_mode
        }
        new_annotations.append(new_annotation)

    return np.array(target_image), new_annotations
