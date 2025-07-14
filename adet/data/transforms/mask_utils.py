import os.path as osp
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class MaskApplier:
    def __init__(self, cfg):
        self.annotation_root = cfg.DATASETS.ANNOTATION_ROOT
        self.train_annotation_dir = osp.join(self.annotation_root, "train")
        self.test_annotation_dir = osp.join(self.annotation_root, "test")

    def get_annotation_path(self, image_number, is_train):
        annotation_dir = self.train_annotation_dir if is_train else self.test_annotation_dir
        return osp.join(annotation_dir, f"gt_img_{image_number}.txt")

    def apply_black_mask(self, image, dataset_dict, is_train=True):
        image = image.copy()
        image_file_name = dataset_dict["file_name"]
        image_number = osp.splitext(osp.basename(image_file_name))[0].split('_')[-1]
        annotation_path = self.get_annotation_path(image_number, is_train)

        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            coords, label = self._parse_annotation_line(line)
            if coords is not None and label == '###':
                self._apply_mask_to_region(image, coords)
                
        return image, dataset_dict

    def _parse_annotation_line(self, line):
        data = line.strip().split(',')
        if len(data) < 8 or any(not d.strip() for d in data[:8]):
            return None, None
        
        try:
            coords = list(map(int, data[:8]))
            label = ','.join(data[8:])
            return coords, label
        except ValueError:
            return None, None

    def _apply_mask_to_region(self, image, coords):
        x_coords = coords[::2]
        y_coords = coords[1::2]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        image[int(y_min):int(y_max), int(x_min):int(x_max)] = 0 
