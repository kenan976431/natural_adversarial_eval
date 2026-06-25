"""
datasets/typographic.py

Two typographic attack scenarios from the paper:
  1. RTA-100  — real-world physically attached labels (download required).
  2. ImageNet-typo — synthetic: wrong class label text overlaid on ImageNet images.

Usage:
    from datasets.typographic import add_typographic_text, ImageNetTypoDataset
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ---------------------------------------------------------------------------
# Primitive: add misleading text to a single PIL image
# ---------------------------------------------------------------------------

def add_typographic_text(
    image: Image.Image,
    text: str,
    font_size: int = 40,
    color: Tuple[int, int, int] = (255, 0, 0),
    position: Optional[Tuple[int, int]] = None,
    font_path: Optional[str] = None,
) -> Image.Image:
    """
    Overlay `text` onto `image` (synthetic typographic attack).

    Args:
        image:      Source PIL image.
        text:       Misleading class label to overlay (e.g. wrong class name).
        font_size:  Font size in pixels.
        color:      Text RGB colour.
        position:   (x, y) top-left corner. If None, placed at centre.
        font_path:  Path to a .ttf font file. Falls back to PIL default.

    Returns:
        New PIL image with text overlaid.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    if position is None:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        position = ((img.width - w) // 2, (img.height - h) // 2)

    draw.text(position, text, fill=color, font=font)
    return img


# ---------------------------------------------------------------------------
# Dataset: ImageNet-typo (synthetic)
# ---------------------------------------------------------------------------

class ImageNetTypoDataset:
    """
    Wraps an ImageNet-style folder dataset and applies synthetic typographic
    attacks on-the-fly: each image gets its label replaced by a random *wrong*
    class name overlaid as text.

    Expects ImageNet folder structure:
        root/
          n01440764/  (synset)
              ILSVRC2012_val_00000293.JPEG
              ...

    Args:
        root:        Path to ImageNet validation set.
        class_names: List of human-readable class names (1000 for ImageNet-1K).
        transform:   Optional torchvision transform applied after text overlay.
    """

    def __init__(self, root: str, class_names: List[str], transform=None):
        self.root = Path(root)
        self.class_names = class_names
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        synset_dirs = sorted(self.root.iterdir())
        for idx, synset_dir in enumerate(synset_dirs):
            if synset_dir.is_dir():
                for img_path in synset_dir.glob("*.JPEG"):
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # Pick a wrong class (guaranteed ≠ true label)
        wrong_classes = [c for i, c in enumerate(self.class_names) if i != label]
        wrong_label = random.choice(wrong_classes)

        image = add_typographic_text(image, text=wrong_label)

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# Dataset: RTA-100 (real-world typographic attacks)
# ---------------------------------------------------------------------------

class RTA100Dataset:
    """
    Real-world Typographic Attack dataset (RTA-100).

    Download from: https://github.com/azuma164/Defense-Prefix
    Expected layout:
        rta100_root/
            images/
                <category>/
                    image_01.jpg ...
            labels.csv   (columns: filename, true_label, attack_label)

    Paper: "Defense-prefix for preventing typographic attacks on CLIP", ICCV 2023.
    """

    def __init__(self, root: str, transform=None):
        import csv
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int, str]] = []

        labels_file = self.root / "labels.csv"
        with open(labels_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = self.root / "images" / row["filename"]
                self.samples.append((img_path, int(row["true_label"]), row["attack_label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, attack_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, attack_label
