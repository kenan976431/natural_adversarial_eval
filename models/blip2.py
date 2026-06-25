"""
models/blip2.py
Wrapper for BLIP-2 (Salesforce/blip2-*) via HuggingFace transformers.
Auto-downloads on first use.
"""

import torch
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2Model
from PIL import Image
from typing import List, Union


class BLIP2Model:
    """
    Wraps BLIP-2 for feature extraction and VQA.
    Note: BLIP-2 is excluded from classification evaluation (paper Sec.3.3).
    """

    def __init__(self, model_id: str = "Salesforce/blip2-flan-t5-xl",
                 device: str = "cuda"):
        self.device   = device
        self.model_id = model_id
        print(f"[BLIP2] Loading {model_id} ...")
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model     = Blip2Model.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device).eval()

    @torch.no_grad()
    def encode_image(self, images: Union[List[Image.Image], torch.Tensor]):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats  = self.model.get_image_features(**inputs).pooler_output
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def answer_question(self, image: Image.Image, question: str) -> str:
        """Run VQA inference for a single image-question pair."""
        from transformers import Blip2ForConditionalGeneration
        if not hasattr(self, "_gen_model"):
            self._gen_model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.float16
            ).to(self.device).eval()
        inputs  = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        out_ids = self._gen_model.generate(**inputs, max_new_tokens=30)
        return self.processor.decode(out_ids[0], skip_special_tokens=True).strip()
