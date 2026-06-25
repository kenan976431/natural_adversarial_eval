#!/usr/bin/env python3
"""
scripts/run_classification.py

Zero-shot image classification evaluation across all natural adversarial
datasets described in the paper.

Examples:
    # Single model, single dataset
    python scripts/run_classification.py \
        --model clip \
        --model-id openai/clip-vit-base-patch16 \
        --dataset imagenet-a \
        --output-dir results/classification/

    # All paper models, all datasets
    python scripts/run_classification.py --all \
        --imagenet-root /data/imagenet/val \
        --imagenet-a-root /data/imagenet-a \
        --rta100-root /data/rta100 \
        --langadv-root datasets/langadv \
        --output-dir results/classification/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model, PAPER_MODELS
from tasks.classification import run_all_classification_benchmarks
from datasets.typographic import ImageNetTypoDataset, RTA100Dataset

# ImageNet-1K class names (loaded from HuggingFace)
def load_imagenet_class_names():
    from datasets import load_dataset
    ds = load_dataset("imagenet-1k", split="validation", streaming=True,
                      trust_remote_code=True)
    return ds.features["label"].names


def get_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def build_dataloaders(args, transform) -> dict:
    dataloaders = {}

    # ImageNet-A (auto-download via HuggingFace)
    if args.dataset in ("imagenet-a", "all"):
        from datasets import load_dataset
        print("[Data] Loading ImageNet-A from HuggingFace ...")
        ds = load_dataset("Benjamin-eecs/imagenet-a", split="test",
                          trust_remote_code=True)
        # Wrap as a simple list dataset
        class HFDataset(torch.utils.data.Dataset):
            def __init__(self, hf_ds, tfm):
                self.ds, self.tfm = hf_ds, tfm
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                item = self.ds[i]
                return self.tfm(item["image"].convert("RGB")), item["label"]
        dataloaders["imagenet-a"] = DataLoader(
            HFDataset(ds, transform), batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True
        )

    # ImageNet-typo (synthetic, generated on-the-fly)
    if args.dataset in ("imagenet-typo", "all") and args.imagenet_root:
        class_names = load_imagenet_class_names()
        typo_ds = ImageNetTypoDataset(args.imagenet_root, class_names, transform)
        dataloaders["imagenet-typo"] = DataLoader(
            typo_ds, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True
        )

    # RTA-100
    if args.dataset in ("rta100", "all") and args.rta100_root:
        rta_ds = RTA100Dataset(args.rta100_root, transform)
        # Wrap to drop the attack_label column
        class Wrap(torch.utils.data.Dataset):
            def __init__(self, ds): self.ds = ds
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                img, lbl, _ = self.ds[i]; return img, lbl
        dataloaders["rta100"] = DataLoader(
            Wrap(rta_ds), batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True
        )

    # LangAdv (pre-generated images in folder per class)
    if args.dataset in ("langadv", "all") and args.langadv_root:
        langadv_ds = datasets.ImageFolder(args.langadv_root, transform=transform)
        dataloaders["langadv"] = DataLoader(
            langadv_ds, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True
        )

    return dataloaders


def run_single(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(args.model, args.model_id, device=device)

    transform   = get_transform()
    class_names = load_imagenet_class_names()
    dataloaders = build_dataloaders(args, transform)

    if not dataloaders:
        print("[ERROR] No datasets found. Check your --dataset flag and data paths.")
        sys.exit(1)

    results = run_all_classification_benchmarks(
        model, dataloaders, class_names, device=device
    )

    out_dir = Path(args.output_dir) / args.model / args.model_id.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "classification_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Results saved to {out_file}")


def run_all(args):
    """Reproduce Table in paper: all 22 models × all datasets."""
    all_results = {}
    for family, model_ids in PAPER_MODELS.items():
        for model_id in model_ids:
            print(f"\n{'='*60}")
            print(f"Model: {family} / {model_id}")
            print(f"{'='*60}")
            args.model    = family
            args.model_id = model_id
            args.dataset  = "all"
            run_single(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot classification evaluation")
    p.add_argument("--model",          type=str, default="clip",
                   choices=["clip", "robust_clip", "siglip2", "blip2", "paligemma2"])
    p.add_argument("--model-id",       type=str,
                   default="openai/clip-vit-base-patch16")
    p.add_argument("--dataset",        type=str, default="imagenet-a",
                   choices=["imagenet-a", "imagenet-typo", "rta100", "langadv", "all"])
    p.add_argument("--all",            action="store_true",
                   help="Run all paper models on all datasets")
    p.add_argument("--imagenet-root",  type=str, default=None,
                   help="Path to ImageNet-1K validation folder")
    p.add_argument("--imagenet-a-root",type=str, default=None,
                   help="Path to ImageNet-A (auto-downloads if omitted)")
    p.add_argument("--rta100-root",    type=str, default=None)
    p.add_argument("--langadv-root",   type=str, default="datasets/langadv")
    p.add_argument("--output-dir",     type=str, default="results/classification")
    p.add_argument("--batch-size",     type=int, default=64)
    p.add_argument("--num-workers",    type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        run_all(args)
    else:
        run_single(args)
