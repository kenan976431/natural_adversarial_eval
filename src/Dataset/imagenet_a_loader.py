"""
ImageNet-A loader
from JSON file path
"""

import os
import json
from PIL import Image
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader


# ImageNet-A contains 200 classes (subset of ImageNet-1K)

IMAGENET_A_TEMPLATES = [
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


def load_imagenet_a_classes(json_path: str) -> List[str]:
    """
    Load ImageNet-A classes from samples.json
    Extract unique class labels from ground_truth field
    
    Args:
        json_path: Path to samples.json file
        
    Returns:
        List of unique class names (sorted)
    """
    if not os.path.exists(json_path):
        print(f"Warning: JSON file not found: {json_path}")
        return [f"class_{i}" for i in range(200)]
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Extract samples list
    if isinstance(json_data, dict) and 'samples' in json_data:
        samples = json_data['samples']
    elif isinstance(json_data, list):
        samples = json_data
    else:
        print("Warning: Invalid JSON format")
        return [f"class_{i}" for i in range(200)]
    
    # Collect unique class labels
    class_set = set()
    for sample in samples:
        label = sample.get('ground_truth', {}).get('label', '')
        if label:
            class_set.add(label)
    
    # Sort classes alphabetically for consistent ordering
    classes = sorted(list(class_set))
    
    print(f"Loaded {len(classes)} unique classes from JSON")
    
    return classes


class ImageNetAParquetDataset(Dataset):
    """
    Load ImageNet-A dataset from JSON file
    """
    
    def __init__(
        self,
        data_dir: str,
        json_file: str = "samples.json",
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: Root data path containing JSON file and images
            json_file: Name of JSON file (default: "samples.json")
            transform: Image preprocessing function
            max_samples: Maximum number of samples to load
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples = max_samples
        
        # Load JSON file
        json_path = os.path.join(data_dir, json_file)
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"JSON file not found: {json_path}"
            )
        
        print(f"Loading ImageNet-A from {json_path}")
        
        # Load class names from JSON file
        self.classes = load_imagenet_a_classes(json_path)
        self.num_classes = len(self.classes)
        
        # Load data
        self.data = self._load_data(json_path)
        print(f"Loaded {len(self.data)} samples")
    
    def _load_data(self, json_path: str) -> List[Dict]:
        """Load data from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Extract samples list
        if isinstance(json_data, dict) and 'samples' in json_data:
            samples = json_data['samples']
        elif isinstance(json_data, list):
            samples = json_data
        else:
            raise ValueError("JSON file must contain 'samples' key or be a list")
        
        all_data = []
        for sample in samples:
            # Extract relevant information
            data_item = {
                'filepath': sample.get('filepath', ''),
                'label': sample.get('ground_truth', {}).get('label', ''),
                'metadata': sample.get('metadata', {})
            }
            
            all_data.append(data_item)
            
            if self.max_samples and len(all_data) >= self.max_samples:
                break
        
        return all_data
    
    def _load_image(self, filepath: str) -> Image.Image:
        """Load image from filepath"""
        # Construct full path
        full_path = os.path.join(self.data_dir, filepath)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")
        
        return Image.open(full_path).convert('RGB')
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.data[idx]
        
        # Load image
        image = self._load_image(sample['filepath'])
        
        # Get label - convert label name to index
        label_name = sample['label']
        try:
            label = self.classes.index(label_name)
        except ValueError:
            # If label not in classes, try to find it or default to 0
            print(f"Warning: Label '{label_name}' not found in classes")
            label = 0
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label index"""
        if 0 <= label < len(self.classes):
            return self.classes[label]
        return f"unknown_{label}"


def display_image(dataset: ImageNetAParquetDataset, index: int):
    """
    Load and display image, print class
    
    Args:
        dataset: ImageNetAParquetDataset instance
        index: Sample index
    """
    if index >= len(dataset):
        print(f"Error: Index {index} is out of bounds for dataset size {len(dataset)}")
        return
    
    raw_sample = dataset.data[index]
    image = dataset._load_image(raw_sample['filepath'])
    
    image.save('show_image.jpg')
    print(f"Saved image to show_image.jpg")
    print(f"Label: {raw_sample['label']}")


def create_dataloader(
    data_dir: str,
    json_file: str = "samples.json",
    transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, ImageNetAParquetDataset]:
    """
    Create DataLoader for ImageNet-A
    
    Args:
        data_dir: Root data path
        json_file: JSON file name
        transform: Image preprocessing function
        batch_size: Batch size
        num_workers: Number of worker processes
        max_samples: Maximum samples to load
    
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    dataset = ImageNetAParquetDataset(
        data_dir=data_dir,
        json_file=json_file,
        transform=transform,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset
