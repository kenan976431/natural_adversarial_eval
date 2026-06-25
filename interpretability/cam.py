"""
interpretability/cam.py

Transformer Class Activation Mapping (CAM) following Chefer et al. 2021:
    "Generic Attention-model Explainability for Interpreting
     Bi-Modal and Encoder-Decoder Transformers", ICCV 2021.

Implements Eqs. 3–4 from the paper:
    A = E_h( (∇A ◦ A)+ )          (head-aggregated relevance, Eq.3)
    R_qq <- R_qq + A · R_qq        (layer-wise relevance propagation, Eq.4)

Usage:
    cam = TransformerCAM(model.model)        # pass raw ViT
    heatmap = cam.generate(image_tensor, target_class=3)
    cam.visualize(image_pil, heatmap, save_path="cam_out.png")
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional


class TransformerCAM:
    """
    Gradient-weighted attention relevance map for ViT-based vision encoders.

    Args:
        model: A ViT model with accessible .blocks (timm) or
               .vision_model.encoder.layers (HuggingFace) attention layers.
               Must expose intermediate attention weights.
    """

    def __init__(self, model):
        self.model  = model
        self.hooks  = []
        self._attn_maps:  list = []
        self._attn_grads: list = []

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Attach forward+backward hooks to every attention layer."""
        self._attn_maps  = []
        self._attn_grads = []

        def _fwd_hook(module, inp, out):
            # out is typically (attn_output, attn_weights)
            # HuggingFace CLIP returns (context, weights)
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attn = out[1].detach()   # (B, heads, Q, K)
                attn.requires_grad_(True)
                self._attn_maps.append(attn)

        def _bwd_hook(grad):
            self._attn_grads.append(grad)

        for module in self.model.modules():
            cls_name = type(module).__name__
            if "Attention" in cls_name and "SelfAttention" not in cls_name:
                h = module.register_forward_hook(_fwd_hook)
                self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    # ------------------------------------------------------------------
    # CAM generation
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def generate(
        self,
        image_tensor: torch.Tensor,   # (1, C, H, W) pre-processed
        target_class: int,
        text_features: Optional[torch.Tensor] = None,
        patch_size: int = 16,
    ) -> np.ndarray:
        """
        Compute a spatial heatmap for `image_tensor` highlighting regions
        most relevant to predicting `target_class`.

        Args:
            image_tensor:  Pre-processed image tensor (1, C, H, W).
            target_class:  Integer class index for gradient target.
            text_features: (C, D) text embeddings for similarity scoring.
                           If None, uses raw logits from a linear head.
            patch_size:    ViT patch size in pixels (default 16).

        Returns:
            heatmap: (H_patch, W_patch) numpy array in [0, 1].
        """
        self.model.eval()
        self._register_hooks()
        self._attn_maps.clear()
        self._attn_grads.clear()

        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass
        if text_features is not None:
            img_feat = self.model.encode_image(image_tensor)        # (1, D)
            img_feat = F.normalize(img_feat, dim=-1)
            logits   = img_feat @ text_features.T                   # (1, C)
        else:
            logits = self.model(image_tensor)

        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)

        self._remove_hooks()

        # Register backward hooks on stored attn maps
        for attn in self._attn_maps:
            attn.register_hook(lambda g: self._attn_grads.append(g))

        # Build relevance map R (Eqs.3-4)
        num_layers = len(self._attn_maps)
        if num_layers == 0:
            raise RuntimeError("No attention maps captured. Check model architecture.")

        # Initialise relevance as identity (CLS token attends to itself)
        num_tokens = self._attn_maps[0].shape[-1]
        R = torch.eye(num_tokens, device=image_tensor.device)

        for attn, grad in zip(self._attn_maps, self._attn_grads):
            # Eq.3: head-aggregated relevance
            cam = (grad * attn).clamp(min=0).mean(dim=1)  # (B, Q, K) averaged over heads
            cam = cam.squeeze(0)                           # (Q, K)

            # Eq.4: residual relevance propagation
            R = R + cam @ R

        # Extract patch-level relevance (exclude CLS token at index 0)
        patch_relevance = R[0, 1:]          # (num_patches,)
        num_patches     = int(patch_relevance.shape[0] ** 0.5)
        heatmap = patch_relevance.reshape(num_patches, num_patches)

        # Normalise to [0, 1]
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def visualize(
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Overlay a heatmap on the original image (blue=low, red=high).

        Args:
            original_image: PIL Image.
            heatmap:        2D numpy array ∈ [0,1].
            alpha:          Blend factor for overlay.
            colormap:       Matplotlib colormap name.
            save_path:      If provided, save the result to this path.

        Returns:
            Blended PIL image.
        """
        orig_w, orig_h = original_image.size
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.BILINEAR
            )
        ) / 255.0

        cmap   = cm.get_cmap(colormap)
        colored = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
        overlay_img = Image.fromarray(colored)

        blended = Image.blend(original_image.convert("RGB"), overlay_img, alpha=alpha)

        if save_path:
            blended.save(save_path)

        return blended
