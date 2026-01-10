"""
RTA100 Dataset Loader
"""

import os
from PIL import Image
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader


RTA100_TEMPLATES = ["a photo of a {}."]


class RTA100Dataset(Dataset):
    """
    RTA100 dataset loader
    """
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            root: root directory path
            transform: image transform function
            max_samples: maximum number of samples to load
        """
        self.img_dir = data_dir
        self.transform = transform
        self.max_samples = max_samples
        
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Data directory not found: {self.img_dir}")
        
        # Load data
        self.data = self._load_data()
        
        # Extract unique classes
        self.classes = self._extract_classes()
        self.num_classes = len(self.classes)
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Number of classes: {self.num_classes}")
    
    def _load_data(self) -> List[dict]:
        """Load all image files and labels"""
        all_data = []
        
        img_files = sorted(os.listdir(self.img_dir))
        
        for img_file in img_files:
            # Parse filename: label=X_text=Y.jpg
            try:
                label = img_file.split('_')[0].split('=')[1]
                text = img_file.split('_')[1].split('=')[1][:-4]
                
                sample = {
                    'image_path': os.path.join(self.img_dir, img_file),
                    'label': label,
                    'text': text
                }
                all_data.append(sample)
                
                if self.max_samples and len(all_data) >= self.max_samples:
                    break
                    
            except (IndexError, ValueError) as e:
                print(f"Warning: Failed to parse filename {img_file}: {e}")
                continue
        
        return all_data
    
    def _extract_classes(self) -> List[str]:
        """Extract unique class names from data"""
        class_set = set()
        for sample in self.data:
            class_set.add(sample['label'])
            class_set.add(sample['text'])
        return sorted(list(class_set))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.data[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Get label index
        label = self.classes.index(sample['label'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label index"""
        if 0 <= label < len(self.classes):
            return self.classes[label]
        return f"unknown_{label}"


def display_image(dataset: RTA100Dataset, index: int):
    """
    Load and display image, print class
    
    Args:
        dataset: RTA100Dataset instance
        index: sample index
    """
    if index >= len(dataset):
        print(f"Error: Index {index} is out of bounds for dataset size {len(dataset)}")
        return
    
    sample = dataset.data[index]
    image = Image.open(sample['image_path']).convert('RGB')
    
    image.save('show_image.jpg')
    print(f"Label: {sample['label']}, Text: {sample['text']}")


def create_dataloader(
    root: str,
    transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    shuffle: bool = False
) -> Tuple[DataLoader, RTA100Dataset]:
    """
    Create DataLoader for RTA100
    
    Args:
        root: root directory path
        transform: image transform function
        batch_size: batch size
        num_workers: number of workers
        max_samples: maximum samples to load
        shuffle: whether to shuffle data
    
    Returns:
        DataLoader and Dataset
    """
    dataset = RTA100Dataset(
        root=root,
        transform=transform,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset
