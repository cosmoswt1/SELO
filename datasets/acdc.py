"""
ACDC Dataset for DinoFlow Training and Evaluation

ACDC (Adverse Conditions Dataset with Correspondences) contains:
- 4 conditions: fog, night, rain, snow
- Train: 400 images per condition (1600 total)
- Val: ~100 images per condition (406 total)
- Test: 500 images per condition (2000 total)
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class ACDCDataset(Dataset):
    """
    ACDC Dataset for unlabeled training (SFDA setting).

    Args:
        root: Path to ACDC dataset root
        split: 'train', 'val', or 'test'
        conditions: List of conditions to include
        transform: Optional transform function
        resize: Resize shorter side to this value
        crop_size: Random crop size (H, W)
    """

    CONDITIONS = ['fog', 'night', 'rain', 'snow']

    def _find_dir(self, root: Path, name: str) -> Optional[Path]:
        direct = root / name
        if direct.exists():
            return direct
        for p in root.iterdir():
            if not p.is_dir():
                continue
            cand = p / name
            if cand.exists():
                return cand
            for q in p.iterdir():
                if not q.is_dir():
                    continue
                cand2 = q / name
                if cand2.exists():
                    return cand2
        return None

    def __init__(
        self,
        root: str,
        split: str = 'train',
        conditions: List[str] = None,
        transform: Callable = None,
        resize: int = 1024,
        crop_size: Tuple[int, int] = (512, 512)
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.conditions = conditions if conditions else self.CONDITIONS
        self.transform = transform
        self.resize = resize
        self.crop_size = crop_size

        # Validate conditions
        for cond in self.conditions:
            if cond not in self.CONDITIONS:
                raise ValueError(f"Invalid condition: {cond}. Must be one of {self.CONDITIONS}")

        # Collect image paths
        self.images = []
        self.image_conditions = []

        rgb_root = self._find_dir(self.root, 'rgb_anon_trainvaltest')
        if rgb_root is not None:
            rgb_base = rgb_root / 'rgb_anon'
        else:
            rgb_base = self.root / 'rgb_anon'
        print(f"[ACDCDataset] rgb base: {rgb_base}")

        for cond in self.conditions:
            cond_dir = rgb_base / cond / split
            if not cond_dir.exists():
                print(f"Warning: {cond_dir} does not exist")
                continue

            for img_path in sorted(cond_dir.rglob('*_rgb_anon.png')):
                self.images.append(img_path)
                self.image_conditions.append(cond)

        print(f"Loaded {len(self.images)} images from ACDC {split} split")
        for cond in self.conditions:
            count = sum(1 for c in self.image_conditions if c == cond)
            print(f"  {cond}: {count} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.images[idx]
        condition = self.image_conditions[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Resize keeping aspect ratio
        w, h = image.size
        if h < w:
            new_h = self.resize
            new_w = int(w * self.resize / h)
        else:
            new_w = self.resize
            new_h = int(h * self.resize / w)

        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Random crop
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size

            # Pad if needed
            if new_h < crop_h or new_w < crop_w:
                pad_h = max(crop_h - new_h, 0)
                pad_w = max(crop_w - new_w, 0)
                image = TF.pad(image, [0, 0, pad_w, pad_h], padding_mode='reflect')
                new_h, new_w = new_h + pad_h, new_w + pad_w

            # Random crop
            top = torch.randint(0, new_h - crop_h + 1, (1,)).item()
            left = torch.randint(0, new_w - crop_w + 1, (1,)).item()
            image = TF.crop(image, top, left, crop_h, crop_w)

        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)

        # Convert to tensor and normalize
        image = TF.to_tensor(image)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        # Apply additional transform if provided
        if self.transform is not None:
            image = self.transform(image)

        return {
            'image': image,
            'condition': condition,
            'path': str(img_path)
        }


class ACDCEvalDataset(Dataset):
    """
    ACDC Dataset for evaluation with ground truth labels.

    Args:
        root: Path to ACDC dataset root
        split: 'val' or 'test'
        conditions: List of conditions to include
        resize: Resize shorter side to this value
    """

    CONDITIONS = ['fog', 'night', 'rain', 'snow']

    def _find_dir(self, root: Path, name: str) -> Optional[Path]:
        direct = root / name
        if direct.exists():
            return direct
        for p in root.iterdir():
            if not p.is_dir():
                continue
            cand = p / name
            if cand.exists():
                return cand
            for q in p.iterdir():
                if not q.is_dir():
                    continue
                cand2 = q / name
                if cand2.exists():
                    return cand2
        return None

    def __init__(
        self,
        root: str,
        split: str = 'val',
        conditions: List[str] = None,
        resize: int = 1024
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.conditions = conditions if conditions else self.CONDITIONS
        self.resize = resize

        # Collect image and label paths
        self.images = []
        self.labels = []
        self.image_conditions = []

        rgb_root = self._find_dir(self.root, 'rgb_anon_trainvaltest')
        gt_root = self._find_dir(self.root, 'gt_trainval')
        rgb_base = (rgb_root / 'rgb_anon') if rgb_root is not None else (self.root / 'rgb_anon')
        gt_base = (gt_root / 'gt') if gt_root is not None else (self.root / 'gt')
        print(f"[ACDCEvalDataset] rgb base: {rgb_base}")
        print(f"[ACDCEvalDataset] gt base: {gt_base}")

        for cond in self.conditions:
            cond_rgb_dir = rgb_base / cond / split
            cond_gt_dir = gt_base / cond / split

            if not cond_rgb_dir.exists():
                print(f"Warning: {cond_rgb_dir} does not exist")
                continue

            for img_path in sorted(cond_rgb_dir.rglob('*_rgb_anon.png')):
                # Find corresponding GT
                rel_path = img_path.relative_to(cond_rgb_dir)
                gt_name = img_path.name.replace('_rgb_anon.png', '_gt_labelTrainIds.png')
                gt_path = cond_gt_dir / rel_path.parent / gt_name

                if gt_path.exists():
                    self.images.append(img_path)
                    self.labels.append(gt_path)
                    self.image_conditions.append(cond)

        print(f"Loaded {len(self.images)} images from ACDC {split} split (with GT)")
        for cond in self.conditions:
            count = sum(1 for c in self.image_conditions if c == cond)
            print(f"  {cond}: {count} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.images[idx]
        gt_path = self.labels[idx]
        condition = self.image_conditions[idx]

        # Load image and label
        image = Image.open(img_path).convert('RGB')
        label = Image.open(gt_path)

        # Resize keeping aspect ratio
        w, h = image.size
        if h < w:
            new_h = self.resize
            new_w = int(w * self.resize / h)
        else:
            new_w = self.resize
            new_h = int(h * self.resize / w)

        image = image.resize((new_w, new_h), Image.BILINEAR)
        label = label.resize((new_w, new_h), Image.NEAREST)

        # Pad to be divisible by 32
        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32

        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], padding_mode='reflect')
            label = TF.pad(label, [0, 0, pad_w, pad_h], fill=255)

        # Convert to tensor
        image = TF.to_tensor(image)
        label = torch.from_numpy(np.array(label)).long()

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return {
            'image': image,
            'label': label,
            'condition': condition,
            'path': str(img_path),
            'original_size': (h, w),
            'padded_size': (new_h + pad_h, new_w + pad_w)
        }


def get_acdc_class_names() -> List[str]:
    """Get ACDC/Cityscapes class names."""
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ]


def get_acdc_class_colors() -> np.ndarray:
    """Get ACDC/Cityscapes class colors for visualization."""
    return np.array([
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ], dtype=np.uint8)
