"""
datasets/langadv.py

Language-Induced Adversarial Example Generation.

Implements the adaptive genetic algorithm from:
    Zhu et al., "Natural Language Induced Adversarial Images", ACM MM 2024.
    https://dl.acm.org/doi/10.1145/3664647.3688991

The algorithm optimises discrete text prompts (describing animal scenes)
to maximise Attack Success Rate (ASR) on a target CLIP-based classifier
while preserving semantic consistency (CLIP similarity to ground-truth prompt).

Pipeline:
  1. Initialise a population of prompts (20 per class).
  2. Generate images with Z-Image (or compatible generator).
  3. Evaluate ASR and semantic consistency.
  4. Evolve over 8 generations (crossover + word-level mutation).
  5. Save final adversarial images (160 per class).

Usage:
    python scripts/generate_langadv.py \
        --classes cat dog bird horse cow \
        --output-dir datasets/langadv/
"""

import random
import copy
import json
from pathlib import Path
from typing import List, Tuple, Callable

import torch
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Abstract image generator interface
# ---------------------------------------------------------------------------

class BaseImageGenerator:
    """Override this class to plug in any text-to-image backend."""

    def generate(self, prompt: str, num_images: int = 1) -> List[Image.Image]:
        raise NotImplementedError


class ZImageGenerator(BaseImageGenerator):
    """
    Calls the Z-Image API (arXiv:2511.22699).
    Replace `api_url` and `api_key` with your credentials.
    """

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def generate(self, prompt: str, num_images: int = 1) -> List[Image.Image]:
        import requests, base64, io
        payload = {"prompt": prompt, "n": num_images}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        images = []
        for item in resp.json()["images"]:
            img_bytes = base64.b64decode(item["b64_json"])
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        return images


# ---------------------------------------------------------------------------
# Genetic algorithm core
# ---------------------------------------------------------------------------

class LangAdvGenerator:
    """
    Generates language-induced adversarial images for a list of animal classes.

    Args:
        classifier:  Callable (images) -> (N,) predicted class indices (int).
        sem_encoder: Callable (images, texts) -> (N,) similarity scores ∈ [-1,1].
                     Should be CLIP cosine similarity.
        img_gen:     BaseImageGenerator instance.
        class_names: List of target class names (e.g. ["cat", "dog", ...]).
        num_variants:   Initial population size per class (paper: 20).
        num_generations: Generations of evolution (paper: 8).
        mutation_prob:   Per-word mutation probability (paper: 0.01).
        images_per_class: Final images to keep per class (paper: 160).
        asr_threshold:   Minimum ASR to keep a prompt (paper: 0.84).
    """

    # Seed adjective/scene banks for prompt initialisation
    _ADJECTIVES = [
        "foggy", "humid", "rainy", "snowy", "dusty", "bright", "dark",
        "crowded", "empty", "ancient", "modern", "colourful", "muddy",
    ]
    _ACTIONS = [
        "stretching", "sleeping", "running", "jumping", "playing",
        "eating", "resting", "looking around", "sitting quietly",
    ]
    _CONTEXTS = [
        "in a park", "on a rooftop", "near a river", "in tall grass",
        "on a busy street", "in a foggy forest", "inside a barn",
    ]

    def __init__(
        self,
        classifier: Callable,
        sem_encoder: Callable,
        img_gen: BaseImageGenerator,
        class_names: List[str],
        num_variants: int = 20,
        num_generations: int = 8,
        mutation_prob: float = 0.01,
        images_per_class: int = 160,
        asr_threshold: float = 0.84,
    ):
        self.classifier    = classifier
        self.sem_encoder   = sem_encoder
        self.img_gen       = img_gen
        self.class_names   = class_names
        self.num_variants  = num_variants
        self.num_generations = num_generations
        self.mutation_prob = mutation_prob
        self.images_per_class = images_per_class
        self.asr_threshold = asr_threshold

    # ------------------------------------------------------------------
    # Prompt initialisation
    # ------------------------------------------------------------------

    def _init_population(self, class_name: str) -> List[str]:
        """Seed `num_variants` random prompts for `class_name`."""
        prompts = []
        for _ in range(self.num_variants):
            n_adj  = random.randint(1, 3)
            adj    = random.sample(self._ADJECTIVES, k=n_adj)
            action = random.choice(self._ACTIONS)
            ctx    = random.choice(self._CONTEXTS)
            count  = random.choice(["a", "two", "three"])
            prompt = f"{count} {' '.join(adj)} {class_name}s {action} {ctx}"
            prompts.append(prompt)
        return prompts

    # ------------------------------------------------------------------
    # Fitness evaluation (Eq.1 + Eq.2 from the paper)
    # ------------------------------------------------------------------

    def _evaluate(
        self, prompts: List[str], true_class_idx: int, gt_prompt: str
    ) -> List[Tuple[str, float, float]]:
        """
        For each prompt: generate images, compute ASR and SEM.
        Returns list of (prompt, asr, sem).
        """
        results = []
        for prompt in prompts:
            images = self.img_gen.generate(prompt, num_images=4)
            if not images:
                continue

            # ASR: fraction of generated images misclassified (Eq.1)
            preds = self.classifier(images)
            asr = float((preds != true_class_idx).float().mean())

            # Semantic consistency: CLIP similarity to gt_prompt (Eq.2)
            sem_scores = self.sem_encoder(images, [gt_prompt] * len(images))
            sem = float(sem_scores.mean())

            results.append((prompt, asr, sem))
        return results

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    @staticmethod
    def _crossover(p1: str, p2: str) -> str:
        """Single-point word-level crossover."""
        words1, words2 = p1.split(), p2.split()
        cut = random.randint(1, max(1, min(len(words1), len(words2)) - 1))
        return " ".join(words1[:cut] + words2[cut:])

    def _mutate(self, prompt: str, class_name: str) -> str:
        """Per-word replacement with probability mutation_prob."""
        words = prompt.split()
        pool  = self._ADJECTIVES + self._ACTIONS + self._CONTEXTS + [class_name]
        new_words = [
            random.choice(pool) if random.random() < self.mutation_prob else w
            for w in words
        ]
        return " ".join(new_words)

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    def generate_class(
        self, class_name: str, true_class_idx: int, output_dir: Path
    ) -> List[Path]:
        """Run the full genetic loop for one class. Returns saved image paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        gt_prompt = f"a photo of a {class_name}"

        population = self._init_population(class_name)
        best_prompts = []

        for gen in range(self.num_generations):
            print(f"  [Gen {gen+1}/{self.num_generations}] evaluating {len(population)} prompts ...")
            scored = self._evaluate(population, true_class_idx, gt_prompt)

            # Keep prompts above ASR threshold, sorted by semantic consistency
            survivors = [(p, asr, sem) for p, asr, sem in scored if asr >= self.asr_threshold]
            survivors.sort(key=lambda x: x[2], reverse=True)
            best_prompts = [p for p, _, _ in survivors]

            if not best_prompts:
                print(f"  [Gen {gen+1}] No survivors above ASR threshold {self.asr_threshold}")
                best_prompts = [s[0] for s in scored[:self.num_variants // 2]]

            # Reproduce next generation
            next_gen = list(best_prompts)
            while len(next_gen) < self.num_variants:
                p1, p2 = random.sample(best_prompts, k=2)
                child  = self._crossover(p1, p2)
                child  = self._mutate(child, class_name)
                next_gen.append(child)
            population = next_gen[:self.num_variants]

        # Final image collection from best prompts
        saved_paths = []
        for prompt in best_prompts[:self.images_per_class // 4]:
            images = self.img_gen.generate(prompt, num_images=4)
            for i, img in enumerate(images):
                fname = output_dir / f"{class_name}_{len(saved_paths):04d}.png"
                img.save(fname)
                saved_paths.append(fname)
                if len(saved_paths) >= self.images_per_class:
                    break
            if len(saved_paths) >= self.images_per_class:
                break

        # Save prompt log
        prompt_log = output_dir / f"{class_name}_prompts.json"
        with open(prompt_log, "w") as f:
            json.dump(best_prompts, f, indent=2)

        print(f"  Saved {len(saved_paths)} images for '{class_name}' -> {output_dir}")
        return saved_paths

    def generate_all(self, output_root: str) -> dict:
        """Run generation for all classes."""
        output_root = Path(output_root)
        results = {}
        for idx, class_name in enumerate(self.class_names):
            print(f"\n[LangAdv] Class {idx+1}/{len(self.class_names)}: {class_name}")
            paths = self.generate_class(
                class_name=class_name,
                true_class_idx=idx,
                output_dir=output_root / class_name,
            )
            results[class_name] = [str(p) for p in paths]
        return results
