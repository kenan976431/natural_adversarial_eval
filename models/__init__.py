"""
models/__init__.py
Registry for all supported vision-language model families.
"""

from .clip_model import CLIPModel
from .robust_clip import RobustCLIPModel
from .siglip2 import SigLIP2Model
from .blip2 import BLIP2Model
from .paligemma2 import PaLiGemma2Model

# Map of family name -> wrapper class
MODEL_REGISTRY = {
    "clip":       CLIPModel,
    "robust_clip": RobustCLIPModel,
    "siglip2":    SigLIP2Model,
    "blip2":      BLIP2Model,
    "paligemma2": PaLiGemma2Model,
}

# All 22 model IDs evaluated in the paper
PAPER_MODELS = {
    "clip": [
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
    ],
    "clip_resnet": [
        "timm/resnet50-clip",           # loaded via open_clip
    ],
    "robust_clip": [
        # FARE variants — download from https://github.com/chs20/RobustVLM
        "robust_clip/ViT-B-16-fare4",
        "robust_clip/ViT-B-32-fare4",
        "robust_clip/ViT-L-14-fare2",
        "robust_clip/ViT-L-14-fare4",
        # TeCoA variants
        "robust_clip/ViT-B-16-tecoa4",
        "robust_clip/ViT-B-32-tecoa4",
        "robust_clip/ViT-L-14-tecoa2",
        "robust_clip/ViT-L-14-tecoa4",
    ],
    "siglip2": [
        "google/siglip2-base-patch16-224",
        "google/siglip2-base-patch16-256",
        "google/siglip2-base-patch16-384",
        "google/siglip2-base-patch16-512",
        "google/siglip2-base-patch32-256",
        "google/siglip2-large-patch16-256",
        "google/siglip2-large-patch16-512",
    ],
    "blip2": [
        "Salesforce/blip2-flan-t5-xl",
        "Salesforce/blip2-flan-t5-xxl",
        "Salesforce/blip2-opt-2.7b",
        "Salesforce/blip2-opt-6.7b",
    ],
    "paligemma2": [
        "google/paligemma2-3b-pt-224",
        "google/paligemma2-3b-pt-448",
        "google/paligemma2-10b-pt-224",
        "google/paligemma2-10b-pt-448",
    ],
}


def build_model(family: str, model_id: str, device: str = "cuda"):
    """
    Factory function: instantiate the correct wrapper by family name.

    Args:
        family:   One of the keys in MODEL_REGISTRY.
        model_id: HuggingFace model ID or local path.
        device:   'cuda' or 'cpu'.

    Returns:
        A model wrapper exposing .encode_image() and .encode_text().
    """
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family '{family}'. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[family](model_id=model_id, device=device)
