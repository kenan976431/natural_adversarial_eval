import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from typing import List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score

import time
from tqdm import tqdm
import numpy as np

from dataset.imagenet_loader import ImageNetParquetDataset, load_imagenet_classes, IMAGENET_TEMPLATES


@dataclass
class EvalMetrics:
    """Container for evaluation metrics"""
    accuracy: float
    top5_accuracy: float
    precision: float
    recall: float
    f1: float
    total_samples: int


class CLIPClassifier:
    """CLIP zero-shot classifier using Transformers library"""
    
    def __init__(
        self,
        model_name: str,
        device: str = 'cuda:0',
        precision: str = 'fp16'
    ):
        """
        Args:
            model_name: HuggingFace model name or local path
            device: device to run on
            precision: 'fp16', 'fp32', 'bf16'
        """
        self.device = device
        self.precision = precision
        self.model_name = model_name
        
        print(f"Loading Transformers CLIP model: {model_name}")
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.model.to(device)
        self.model.eval()
        
        self.text_features = None
        self.class_names = None
    
    def preprocess_images(self, images):
        """Preprocess images using CLIP processor"""
        # images should be PIL images or tensors
        return images
    
    def encode_text_prompts(
        self,
        class_names: List[str],
        templates: List[str] = IMAGENET_TEMPLATES,
        use_ensemble: bool = True
    ) -> torch.Tensor:
        """
        Encode text prompts to features
        
        Args:
            class_names: list of class names
            templates: prompt templates
            use_ensemble: use templates or not
        
        Returns:
            text feature [num_classes, feature_dim]
        """
        self.class_names = class_names
        
        with torch.no_grad():
            if use_ensemble and len(templates) > 1:
                all_features = []
                for class_name in tqdm(class_names, desc="Encoding text"):
                    class_features = []
                    for template in templates:
                        text = template.format(class_name)
                        text_inputs = self.processor(
                            text=[text],
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        )
                        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                        
                        features = self.model.get_text_features(**text_inputs)
                        class_features.append(features)
                    
                    # Average and normalize
                    class_features = torch.stack(class_features).mean(dim=0)
                    class_features = F.normalize(class_features, dim=-1)
                    all_features.append(class_features)
                
                text_features = torch.cat(all_features, dim=0)
            else:
                # Simple template
                texts = [f"a photo of a {name}." for name in class_names]
                text_inputs = self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                text_features = self.model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, dim=-1)
        
        self.text_features = text_features
        return text_features
    
    def classify_batch(
        self,
        images: torch.Tensor,
        temperature: float = 100.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify a batch of images
        
        Args:
            images: [B, C, H, W] tensor or list of PIL images
            temperature: softmax temperature
        
        Returns:
            Predicted class and probability
        """
        if self.text_features is None:
            raise RuntimeError("Text features not encoded. Call encode_text_prompts first.")
        
        with torch.no_grad():
            if self.precision == 'fp16':
                with torch.cuda.amp.autocast():
                    image_features = self.model.get_image_features(images.to(self.device))
            else:
                image_features = self.model.get_image_features(images.to(self.device))
            
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            
            # Calculate similarity
            logits = temperature * image_features @ self.text_features.to(image_features.dtype).T
            probs = logits.softmax(dim=-1)
            predictions = probs.argmax(dim=-1)
        
        return predictions, probs
    
    def evaluate(
        self,
        dataloader: DataLoader,
        class_names: List[str],
        templates: List[str] = IMAGENET_TEMPLATES,
        use_ensemble: bool = True
    ) -> EvalMetrics:
        """
        Evaluate model performance with detailed classification metrics.
        
        Args:
            dataloader: DataLoader containing validation/test data
            class_names: List of class names
            templates: Text prompt templates for zero-shot classification
            use_ensemble: Whether to use ensemble of templates
        
        Returns:
            EvalMetrics object containing comprehensive evaluation metrics
        """
        print("Encoding text prompts...")
        self.encode_text_prompts(class_names, templates, use_ensemble)
        
        num_classes = len(class_names)
        all_predictions = []
        all_labels = []
        all_top5_correct = 0
        
        # Track per-class statistics
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        
        print("Running inference...")
        start_time = time.time()
        
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            predictions, probs = self.classify_batch(images)
            
            # Convert to CPU tensors
            predictions = predictions.cpu()
            labels = labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels)
            
            # Store predictions and labels for metric calculation
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            
            # Calculate Top-5 accuracy
            _, top5_preds = probs.cpu().topk(5, dim=-1)
            for i, label in enumerate(labels):
                if label in top5_preds[i]:
                    all_top5_correct += 1
                
                # Update per-class statistics
                class_total[label.item()] += 1
                if predictions[i] == label:
                    class_correct[label.item()] += 1
        
        total_samples = len(all_labels)
        elapsed_time = time.time() - start_time
        print(f"Inference completed in {elapsed_time:.2f} seconds")
        
        # Convert to numpy arrays for sklearn metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate Top-1 accuracy
        correct = (all_predictions == all_labels).sum()
        accuracy = correct / total_samples
        top5_accuracy = all_top5_correct / total_samples
        
        # Calculate per-class accuracy
        per_class_acc = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                per_class_acc[i] = class_correct[i] / class_total[i]
            else:
                per_class_acc[i] = 0.0

        all_class_ids = list(range(num_classes))
        
        # Calculate precision, recall, and F1 scores using sklearn
        # Macro-averaged metrics (unweighted mean across classes)
        precision = precision_score(all_labels, all_predictions, labels=all_class_ids, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, labels=all_class_ids, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, labels=all_class_ids, average='macro', zero_division=0)
        
        return EvalMetrics(
            accuracy=accuracy,
            top5_accuracy=top5_accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            total_samples=total_samples
        )


def EvaluateCLIP(
    model_name: str,
    data_dir: str,
    split: str = "validation",
    device: str = "cuda:0",
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    class_file: Optional[str] = None,
    use_ensemble: bool = True
) -> EvalMetrics:
    """
    Evaluate CLIP model using Transformers library
    
    Args:
        model_name: HuggingFace model name or local path
        data_dir: Path to ImageNet data
        split: 'validation' or 'train'
        device: Device to run on
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        max_samples: Maximum number of samples to evaluate (None for all)
        class_file: Path to class names file
        use_ensemble: Whether to use ensemble of templates
    
    Returns:
        EvalMetrics with all evaluation results
    """
    classifier = CLIPClassifier(
        model_name=model_name,
        device=device
    )

    class_names = load_imagenet_classes(class_file)

    def clip_transform(image):
        return classifier.processor.image_processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
    
    # Note: You may need to adjust the dataset to work with CLIP processor
    # The processor expects PIL images or already preprocessed tensors
    dataset = ImageNetParquetDataset(
        data_dir=data_dir,
        split=split,
        transform=clip_transform,
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

    for i in range(5):
        img, label = dataset[i]
        class_name = dataset.get_class_name(label)
        print(f"Index {i}: Label ID={label}, Name={class_name}")
    
    metrics = classifier.evaluate(
        dataloader=dataloader,
        class_names=class_names,
        templates=IMAGENET_TEMPLATES,
        use_ensemble=use_ensemble
    )
    
    return metrics


if __name__ == "__main__":
    # Example usage with HuggingFace model
    metrics = EvaluateCLIP(
        model_name='models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3',
        data_dir='datasets--ILSVRC--imagenet-1k/snapshots/49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data',
        split='typo-validation',
        device='cuda:0',
        batch_size=64,
        class_file="Dataset/imagenet_classes.txt",
        use_ensemble=False
    )
    
    print(f"Total Samples: {metrics.total_samples}")
    print(f"Top-1 Accuracy: {metrics.accuracy:.6f}")
    print(f"Top-5 Accuracy: {metrics.top5_accuracy:.6f}")
    print(f"Precision: {metrics.precision:.6f}")
    print(f"Recall: {metrics.recall:.6f}")
    print(f"F1: {metrics.f1:.6f}")
