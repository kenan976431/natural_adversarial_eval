"""
models/clip_model.py
Wrapper around OpenAI CLIP and timm ResNet-CLIP via open_clip.
Auto-downloads from HuggingFace on first use.
"""

import torch
import open_clip
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
from PIL import Image
from typing import List, Union


class CLIPModel:
    """
    Unified wrapper for OpenAI CLIP variants (ViT-B/16, ViT-B/32, ViT-L/14)
    and timm ResNet50-CLIP.

    Exposes:
        encode_image(images)  -> (N, D) float tensor, L2-normalised
        encode_text(texts)    -> (N, D) float tensor, L2-normalised
        logit_scale           -> scalar (learnt temperature)
    """

    # Map HuggingFace IDs to open_clip (model_name, pretrained) tuples
    _OPEN_CLIP_MAP = {
        "openai/clip-vit-base-patch16":   ("ViT-B-16",   "openai"),
        "openai/clip-vit-base-patch32":   ("ViT-B-32",   "openai"),
        "openai/clip-vit-large-patch14":  ("ViT-L-14",   "openai"),
        "timm/resnet50-clip":             ("RN50",        "openai"),
    }

    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device
        self.model_id = model_id

        if model_id in self._OPEN_CLIP_MAP:
            arch, pretrained = self._OPEN_CLIP_MAP[model_id]
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(arch)
            self._backend = "open_clip"
        else:
            # Fallback: load from HuggingFace transformers
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = HFCLIPModel.from_pretrained(model_id)
            self._backend = "hf"

        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def encode_image(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        if self._backend == "open_clip":
            if isinstance(images, list):
                images = torch.stack([self.preprocess(img) for img in images])
            images = images.to(self.device)
            feats = self.model.encode_image(images)
        else:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if self._backend == "open_clip":
            tokens = self.tokenizer(texts).to(self.device)
            feats = self.model.encode_text(tokens)
        else:
            inputs = self.processor(text=texts, return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(feats, dim=-1)

    @property
    def logit_scale(self) -> float:
        if self._backend == "open_clip":
            return self.model.logit_scale.exp().item()
        return self.model.logit_scale.exp().item()

    def zero_shot_classify(self, images, class_names: List[str]) -> torch.Tensor:
        """
        Returns (N, C) similarity matrix using the prompt template
        'a photo of a {class_name}.' as in Radford et al. 2021.
        """
        prompts = [f"a photo of a {c}." for c in class_names]
        text_feats = self.encode_text(prompts)      # (C, D)
        img_feats  = self.encode_image(images)       # (N, D)
        return (self.logit_scale * img_feats @ text_feats.T)  # (N, C)
