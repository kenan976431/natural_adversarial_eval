"""
Animal-10 loader
from image file paths with labels in filename
"""

import os
import glob
from PIL import Image
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader


# Animal-10 class names
ANIMAL_CLASSES = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel"
]

ANIMAL_TEMPLATES = [
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


def load_animal_classes(class_list: Optional[List[str]] = None) -> List[str]:
    """
    Load Animal-10 classes
    
    Args:
        class_list: Optional custom class list
        
    Returns:
        class names (text) [list]
    """
    if class_list is not None and len(class_list) > 0:
        return class_list
    
    return ANIMAL_CLASSES


class AnimalParquetDataset(Dataset):
    """
    Load Animal-10 dataset from directory structure
    Format: animal-10/{class_name}/{id}_{true_label}_as_{pred_label}.png
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
            data_dir: Path to animal-10 directory
            split: Not used for animal-10, kept for compatibility
            transform: Image processing function
            max_samples: Maximum number of samples to load
            class_file: Not used for animal-10, kept for compatibility
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
        # Load classes
        self.classes = load_animal_classes()
        self.num_classes = len(self.classes)
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.data = self._load_data()
        print(f"Loaded {len(self.data)} samples from Animal-10 dataset")
    
    def _load_data(self) -> List[Dict]:
        """
        Load all image paths and extract labels from filenames
        Filename format: {id}_{true_label}_as_{pred_label}.png
        """
        all_data = []
        
        # Iterate through each class directory
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found, skipping...")
                continue
            
            # Find all PNG files in the class directory
            image_files = glob.glob(os.path.join(class_dir, "*.png"))
            
            for img_path in image_files:
                # Extract filename without extension
                filename = os.path.basename(img_path)
                filename_no_ext = os.path.splitext(filename)[0]
                
                # Parse filename: {id}_{true_label}_as_{pred_label}
                # Example: 000_butterfly_as_horse
                parts = filename_no_ext.split('_')
                
                if len(parts) >= 4 and parts[-2] == 'as':
                    # Extract true label (everything between id and '_as_')
                    true_label = '_'.join(parts[1:-2])
                    
                    # Get label index
                    if true_label in self.class_to_idx:
                        label_idx = self.class_to_idx[true_label]
                        
                        sample = {
                            'image_path': img_path,
                            'label': label_idx,
                            'class_name': true_label
                        }
                        all_data.append(sample)
                        
                        if self.max_samples and len(all_data) >= self.max_samples:
                            return all_data
                    else:
                        print(f"Warning: Unknown class '{true_label}' in {filename}")
                else:
                    print(f"Warning: Cannot parse filename {filename}")
        
        return all_data
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from file path"""
        return Image.open(image_path).convert('RGB')
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.data[idx]
        
        image = self._load_image(sample['image_path'])
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label index"""
        if 0 <= label < len(self.classes):
            return self.classes[label]
        return f"unknown_{label}"


def display_image(dataset: AnimalParquetDataset, index: int):
    """
    Load and display image, print class information
    
    Args:
        dataset: AnimalParquetDataset instance
        index: Sample index
    """
    if index >= len(dataset):
        print(f"Error: Index {index} is out of bounds for dataset size {len(dataset)}")
        return
    
    sample = dataset.data[index]
    image: Image.Image = dataset._load_image(sample['image_path'])
    
    print(f"Class: {sample['class_name']} (label: {sample['label']})")
    print(f"Image path: {sample['image_path']}")
    
    image.save('sample.jpg')


def create_dataloader(
    data_dir: str,
    split: str = "validation",
    transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    class_file: Optional[str] = None
) -> Tuple[DataLoader, AnimalParquetDataset]:
    """
    Create DataLoader for Animal-10 dataset
    
    Args:
        data_dir: Path to animal-10 directory
        split: Not used for animal-10, kept for compatibility
        transform: Image transformation function
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_samples: Maximum samples to load
        class_file: Not used for animal-10, kept for compatibility
    
    Returns:
        DataLoader and Dataset instance
    """
    dataset = AnimalParquetDataset(
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
