import torch
from torch.utils.data import DataLoader
from typing import List, Optional
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoProcessor, AutoModel

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


class SigLIP2Classifier:
    """SigLIP2 Zero-shot Classifier (Optimized Feature-based)"""
    
    def __init__(
        self,
        model_name: str,
        device: str = 'cuda:1'
    ):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading SigLIP2 model: {model_name}")
        # Load SigLIP2 model
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # [Num_Classes, Hidden_Dim]
        self.classifier_weights = None 

    def build_classifier(
        self,
        class_names: List[str],
        templates: List[str] = IMAGENET_TEMPLATES
    ):
        """
        Build Zero-shot Classifier
        """
        print(f"Building classifier for {len(class_names)} classes...")
        
        if not templates:
            templates = ["a photo of a {}."]
            
        all_text_features = []
        
        batch_size = 100 
        
        with torch.no_grad():
            for i in tqdm(range(0, len(class_names), batch_size), desc="Encoding Texts"):
                batch_names = class_names[i : i + batch_size]
                
                # get first template
                current_texts = [templates[0].format(name) for name in batch_names]
                
                # text processor
                inputs = self.processor(
                    text=current_texts,
                    padding="max_length", 
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # get text feature
                text_outputs = self.model.text_model(**inputs)
                text_embeds = text_outputs.pooler_output
                
                # norm (text and image features are not pre-aligned)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                all_text_features.append(text_embeds)
                
        # cat features of all classes
        self.classifier_weights = torch.cat(all_text_features, dim=0)
        
        # [Dim, Num_Classes]
        self.classifier_weights = self.classifier_weights.t()

    def evaluate(
        self, 
        dataloader, 
        class_names, 
        templates=None, 
        use_ensemble=False
    ) -> EvalMetrics:
        """
        Evaluate the classifier on a dataset
        
        Args:
            dataloader: DataLoader for evaluation
            class_names: List of class names
            templates: List of text templates
            use_ensemble: Whether to use template ensembling (currently not implemented)
        
        Returns:
            EvalMetrics object with all evaluation metrics
        """
        # Classifier
        if templates is None:
            templates = ["a photo of a {}."]
        self.build_classifier(class_names, templates)
        
        all_predictions = []
        all_labels = []
        top1_correct = 0
        top5_correct = 0
        total_samples = 0
        
        print("Running Inference...")
        
        for batch_data in tqdm(dataloader, desc="Eval"):
            images, labels = batch_data
                
            # image process
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # get image features
                image_outputs = self.model.vision_model(**inputs)
                image_embeds = image_outputs.pooler_output
                
                # norm
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                
                # cal Logits (image @ text)
                logits = torch.matmul(image_embeds, self.classifier_weights)
                
                # add scale and bias
                if hasattr(self.model, "logit_scale"):
                    logits = logits * self.model.logit_scale.exp()
                if hasattr(self.model, "logit_bias"):
                    logits = logits + self.model.logit_bias
            
            probs = logits.softmax(dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            
            # Labels
            if isinstance(labels, torch.Tensor):
                labels_np = labels.numpy()
            else:
                labels_np = np.array(labels)
            
            all_predictions.extend(preds)
            all_labels.extend(labels_np)
            
            top1_correct += (preds == labels_np).sum()
            
            # Top-5
            top5_indices = torch.topk(logits, k=5, dim=1).indices.cpu().numpy()
            for i, label in enumerate(labels_np):
                if label in top5_indices[i]:
                    top5_correct += 1
            
            total_samples += len(labels_np)
        
        accuracy = top1_correct / total_samples
        top5_accuracy = top5_correct / total_samples
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        all_class_ids = list(range(len(class_names)))
        
        # cal Macro-averaged metrics (unweighted mean across classes)
        precision = precision_score(
            all_labels, 
            all_predictions, 
            labels=all_class_ids, 
            average='macro', 
            zero_division=0
        )
        recall = recall_score(
            all_labels, 
            all_predictions, 
            labels=all_class_ids, 
            average='macro', 
            zero_division=0
        )
        f1 = f1_score(
            all_labels, 
            all_predictions, 
            labels=all_class_ids, 
            average='macro', 
            zero_division=0
        )
        
        return EvalMetrics(
            accuracy=accuracy,
            top5_accuracy=top5_accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            total_samples=total_samples
        )
    

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    import torch
    labels = torch.tensor(labels)
    
    return images, labels


def EvaluateSigLIP2(
    model_name: str,
    data_dir: str = None,
    split: str = "validation",
    device: str = "cuda:1",
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    class_file: Optional[str] = None,
    use_ensemble: bool = True
) -> EvalMetrics:
    """
    Evaluate SigLIP2 on ImageNet dataset
    
    Args:
        model_name: HuggingFace model name or local path
        data_dir: Path to ImageNet parquet data
        split: Dataset split ('validation' or 'train')
        device: Device for inference
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        max_samples: Maximum number of samples to evaluate (None for all)
        class_file: Path to class names file
        use_ensemble: Whether to use template ensembling
    
    Returns:
        EvalMetrics object with evaluation results
    """
    
    classifier = SigLIP2Classifier(
        model_name=model_name,
        device=device
    )

    class_names = load_imagenet_classes(class_file)
    
    dataset = ImageNetParquetDataset(
        data_dir=data_dir,
        split=split,
        transform=None,
        max_samples=max_samples,
        class_file=class_file
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    metrics = classifier.evaluate(
        dataloader=dataloader,
        class_names=class_names,
        use_ensemble=use_ensemble
    )
    
    return metrics


if __name__ == "__main__":
    metrics = EvaluateSigLIP2(
        model_name='models--google--siglip2-base-patch16-512/snapshots/a89f5c5093f902bf39d3cd4d81d2c09867f0724b',
        data_dir='datasets--ILSVRC--imagenet-1k/snapshots/49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data',
        split='typo-validation',
        device='cuda:0',
        batch_size=64,
        class_file="Dataset/imagenet_classes.txt",
        use_ensemble=True
    )
    
    print(f"Total Samples: {metrics.total_samples}")
    print(f"Top-1 Accuracy: {metrics.accuracy:.6f}")
    print(f"Top-5 Accuracy: {metrics.top5_accuracy:.6f}")
    print(f"Precision: {metrics.precision:.6f}")
    print(f"Recall: {metrics.recall:.6f}")
    print(f"F1: {metrics.f1:.6f}")
