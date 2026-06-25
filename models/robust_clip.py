"""
models/robust_clip.py
Wrapper for Robust CLIP (FARE and TeCoA variants).

Checkpoints: https://github.com/chs20/RobustVLM
Download and place under: checkpoints/robust_clip/

Model mapping (checkpoint filename -> architecture):
    ViT-B-16-fare4    -> ViT-B/16 FARE eps=4/255
    ViT-B-32-fare4    -> ViT-B/32 FARE eps=4/255
    ViT-L-14-fare2    -> ViT-L/14 FARE eps=2/255
    ViT-L-14-fare4    -> ViT-L/14 FARE eps=4/255
    ViT-B-16-tecoa4   -> ViT-B/16 TeCoA eps=4/255
    ViT-B-32-tecoa4   -> ViT-B/32 TeCoA eps=4/255
    ViT-L-14-tecoa2   -> ViT-L/14 TeCoA eps=2/255
    ViT-L-14-tecoa4   -> ViT-L/14 TeCoA eps=4/255
"""

import torch
import open_clip
import torch.nn.functional as F
from pathlib import Path
from typing import List, Union
from PIL import Image

_ARCH_MAP = {
    "ViT-B-16": "ViT-B-16",
    "ViT-B-32": "ViT-B-32",
    "ViT-L-14": "ViT-L-14",
}

CHECKPOINT_DIR = Path("checkpoints/robust_clip")


class RobustCLIPModel:
    """
    Loads a Robust CLIP checkpoint (FARE or TeCoA) from a local directory.

    model_id examples:
        "robust_clip/ViT-B-16-fare4"
        "robust_clip/ViT-L-14-tecoa4"
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device
        variant = model_id.split("/")[-1]   # e.g. "ViT-B-16-fare4"

        # Parse architecture
        arch = next((a for a in _ARCH_MAP if variant.startswith(a)), None)
        if arch is None:
            raise ValueError(f"Cannot infer architecture from model_id: {model_id}")

        ckpt_path = CHECKPOINT_DIR / f"{variant}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Download from https://github.com/chs20/RobustVLM and place in {CHECKPOINT_DIR}"
            )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            _ARCH_MAP[arch], pretrained=str(ckpt_path)
        )
        self.tokenizer = open_clip.get_tokenizer(_ARCH_MAP[arch])
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def encode_image(self, images):
        if isinstance(images, list):
            images = torch.stack([self.preprocess(img) for img in images])
        feats = self.model.encode_image(images.to(self.device))
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]):
        tokens = self.tokenizer(texts).to(self.device)
        feats  = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    def zero_shot_classify(self, images, class_names: List[str]):
        prompts    = [f"a photo of a {c}." for c in class_names]
        text_feats = self.encode_text(prompts)
        img_feats  = self.encode_image(images)
        scale = self.model.logit_scale.exp().item()
        return scale * (img_feats @ text_feats.T)
