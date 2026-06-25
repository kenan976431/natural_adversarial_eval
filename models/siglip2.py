"""
models/siglip2.py
Wrapper for SigLIP2 family (google/siglip2-*).

Key difference from CLIP: SigLIP2 uses sigmoid loss (not softmax contrastive),
so classification uses the learnt logit_scale + logit_bias rather than plain
cosine similarity. See Section 3.3 of the paper.

Auto-downloads from HuggingFace on first use.
"""

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from PIL import Image
from typing import List, Union


class SigLIP2Model:
    """
    Wrapper for SigLIP2 models.

    Exposes the same interface as CLIPModel:
        encode_image(images)  -> (N, D) float tensor, L2-normalised
        encode_text(texts)    -> (N, D) float tensor, L2-normalised
        zero_shot_classify()  -> (N, C) logit tensor (sigmoid-calibrated)
    """

    def __init__(self, model_id: str = "google/siglip2-base-patch16-224",
                 device: str = "cuda"):
        self.device = device
        self.model_id = model_id
        print(f"[SigLIP2] Loading {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def encode_image(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding="max_length", truncation=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def zero_shot_classify(self, images, class_names: List[str]) -> torch.Tensor:
        """
        SigLIP2 classification uses the sigmoid loss formulation with
        learnt logit_scale and logit_bias (no softmax).
        Returns (N, C) logits.
        """
        prompts = [f"a photo of a {c}." for c in class_names]
        img_feats  = self.encode_image(images)   # (N, D)
        text_feats = self.encode_text(prompts)   # (C, D)

        # Retrieve learnt sigmoid parameters
        logit_scale = self.model.logit_scale.exp()
        logit_bias  = self.model.logit_bias if hasattr(self.model, "logit_bias") else 0.0

        logits = logit_scale * (img_feats @ text_feats.T) + logit_bias  # (N, C)
        return logits
