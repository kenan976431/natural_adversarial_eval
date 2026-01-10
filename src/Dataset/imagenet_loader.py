"""
ImageNet-1K loader
from .parquet path
"""

import os
import glob
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import pandas as pd


# # ImageNet-1K labels
IMAGENET_CLASSES = "imagenet_classes.txt"

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}."
]


def load_imagenet_classes(class_file: Optional[str] = None) -> List[str]:
    """
    load ImageNet-1K classes
    
    Args:
        class_file: file path
        
    Returns:
        class names (text) [list]
    """
    if class_file and os.path.exists(class_file):
        with open(class_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines()]
        
        classes = [c for c in classes if c]

        if classes:
            return classes
            
    return [f"class_{i}" for i in range(1000)]


class ImageNetParquetDataset(Dataset):
    """
    load dataset from .parquet path
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "validation",
        transform=None,
        max_samples: Optional[int] = None,
        class_file: Optional[str] = None
    ):
        """
        Args:
            data_dir: data path
            split: 'train', 'test', 'validation'
            transform: image process function
            max_samples
            class_file
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
        # load classes file
        self.classes = load_imagenet_classes(class_file)
        self.num_classes = len(self.classes)
        
        self.parquet_files = sorted(glob.glob(
            os.path.join(data_dir, f"{split}-*.parquet")
        ))
        
        if not self.parquet_files:
            raise FileNotFoundError(
                f"No parquet files found for split '{split}' in {data_dir}"
            )
        
        print(f"Found {len(self.parquet_files)} parquet files for {split} split")
        
        self.data = self._load_data()
        print(f"Loaded {len(self.data)} samples")
    
    def _load_data(self) -> List[Dict]:
        all_data = []
        
        for pq_file in self.parquet_files:
            table = pq.read_table(pq_file)
            df = table.to_pandas()
            
            for idx, row in df.iterrows():
                sample = {
                    'image': row['image'],
                    'label': row['label']
                }
                all_data.append(sample)
                
                if self.max_samples and len(all_data) >= self.max_samples:
                    return all_data
        
        return all_data
    
    def _load_image(self, image_data) -> Image.Image:
        if isinstance(image_data, dict):
            if 'bytes' in image_data:
                return Image.open(BytesIO(image_data['bytes'])).convert('RGB')
            elif 'path' in image_data:
                return Image.open(image_data['path']).convert('RGB')
        elif isinstance(image_data, bytes):
            return Image.open(BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, Image.Image):
            return image_data.convert('RGB')
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.data[idx]
        
        image = self._load_image(sample['image'])
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label: int) -> str:
        if 0 <= label < len(self.classes):
            return self.classes[label]
        return f"unknown_{label}"
    

def display_image(dataset: ImageNetParquetDataset, index: int):
    """
    load and display images, print class
    
    Args:
        dataset: ImageNetParquetDataset
        index
    """
    if index >= len(dataset):
        print(f"Error: Index {index} is out of bounds for dataset size {len(dataset)}")
        return
    
    raw_sample = dataset.data[index]
    image: Image.Image = dataset._load_image(raw_sample['image'])
    
    image.save('show_image.jpg') 


def create_dataloader(
    data_dir: str,
    split: str = "validation",
    transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    class_file: Optional[str] = None
) -> Tuple[DataLoader, ImageNetParquetDataset]:
    """
    Create DataLoader
    
    Args:
        data_dir
        split
        transform
        batch_size
        num_workers
        max_samples
        class_file
    
    Returns:
        DataLoader and Dataset
    """
    dataset = ImageNetParquetDataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
        max_samples=max_samples,
        class_file=class_file
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset
