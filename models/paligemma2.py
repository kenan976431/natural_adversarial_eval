"""
models/paligemma2.py
Wrapper for PaLiGemma2 (google/paligemma2-*) via HuggingFace transformers.
Auto-downloads on first use.

Requires: transformers >= 4.41.0
Access: model is gated — accept license at https://huggingface.co/google/paligemma2-3b-pt-224
"""

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
from typing import List


class PaLiGemma2Model:
    """
    Wraps PaLiGemma2 for VQA and feature extraction.
    Image encoder is SigLIP-So400m; text decoder is Gemma2.
    """

    def __init__(self, model_id: str = "google/paligemma2-3b-pt-224",
                 device: str = "cuda"):
        self.device   = device
        self.model_id = model_id
        print(f"[PaLiGemma2] Loading {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model     = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(device).eval()

    @torch.no_grad()
    def answer_question(self, image: Image.Image, question: str) -> str:
        inputs  = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(out_ids[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]):
        """Extract vision features from the SigLIP image encoder."""
        import torch.nn.functional as F
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        vision_outputs = self.model.vision_tower(**inputs)
        feats = vision_outputs.last_hidden_state.mean(dim=1)
        return F.normalize(feats.float(), dim=-1)
