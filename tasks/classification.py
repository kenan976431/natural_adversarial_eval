"""
tasks/classification.py

Zero-shot image classification evaluation following Radford et al. 2021.
Supports all VLM families through the unified model wrapper interface.

Metrics: Top-1 Accuracy
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict
from tqdm import tqdm


def evaluate_classification(
    model,
    dataloader: DataLoader,
    class_names: List[str],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate a model's zero-shot classification accuracy.

    Args:
        model:       Any model wrapper exposing .zero_shot_classify().
        dataloader:  Yields (images, labels) batches.
        class_names: Full list of class names for prompt construction.
        device:      'cuda' or 'cpu'.

    Returns:
        dict with keys: 'top1_accuracy', 'num_correct', 'num_total'
    """
    model_was_training = False
    if hasattr(model.model, "training") and model.model.training:
        model.model.eval()
        model_was_training = True

    num_correct = 0
    num_total   = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Classifying"):
            labels = labels.to(device)

            logits = model.zero_shot_classify(images, class_names)  # (N, C)
            preds  = logits.argmax(dim=-1)                          # (N,)

            num_correct += (preds == labels).sum().item()
            num_total   += labels.size(0)

    if model_was_training:
        model.model.train()

    acc = num_correct / num_total if num_total > 0 else 0.0
    return {
        "top1_accuracy": acc,
        "num_correct":   num_correct,
        "num_total":     num_total,
    }


def run_all_classification_benchmarks(
    model,
    datasets: Dict[str, DataLoader],
    class_names: List[str],
    device: str = "cuda",
) -> Dict[str, Dict]:
    """
    Convenience wrapper: evaluate across multiple datasets in one call.

    Args:
        datasets: {'imagenet1k': dl, 'imagenet_a': dl, 'imagenet_typo': dl,
                   'rta100': dl, 'langadv': dl}

    Returns:
        Nested dict: {dataset_name: {metric_name: value}}
    """
    results = {}
    for dataset_name, dataloader in datasets.items():
        print(f"\n[Classification] Evaluating on {dataset_name} ...")
        results[dataset_name] = evaluate_classification(
            model, dataloader, class_names, device
        )
        acc = results[dataset_name]["top1_accuracy"]
        print(f"  Top-1 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return results
