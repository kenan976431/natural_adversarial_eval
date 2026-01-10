import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

import time
from tqdm import tqdm
import numpy as np
from PIL import Image

from dataset.vqav2_loader import VQAv2Dataset


@dataclass
class VQAMetrics:
    """Container for VQA evaluation metrics"""
    vqa_accuracy: float
    exact_match: float
    total_samples: int


class BLIP2VQA:
    """BLIP2 Zero-shot VQA Model with FlanT5"""
    
    def __init__(
        self,
        model_name: str,
        device: str = 'cuda:2',
        precision: str = 'fp16',
        max_new_tokens: int = 10,
        num_beams: int = 5,
        length_penalty: float = -1.0
    ):
        self.device = device
        self.precision = precision
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        
        print(f"Loading BLIP2-FlanT5 model: {model_name}")
        print(f"Using prompt format: 'Question: {{q}} Short answer:'")
        print(f"Note: FlanT5 is instruction-tuned and performs significantly better on VQA")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Determine dtype
        if precision == 'fp16':
            dtype = torch.float16
        elif precision == 'bf16':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        self.dtype = dtype
        
        # Load model
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device
        )
        self.model.eval()
        
        print(f"Model loaded successfully on {device}")
        print(f"Generation parameters: num_beams={num_beams}, length_penalty={length_penalty}, max_new_tokens={max_new_tokens}")
    
    def answer_batch(
        self,
        images: List[Image.Image],
        questions: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Generate answers for a batch of image-question pairs
        
        Args:
            images: List of PIL Images
            questions: List of questions
            
        Returns:
            Tuple of (cleaned answers, raw generated texts)
        """
        
        # Format prompts - FlanT5 uses "Question: {} Short answer:" format (from BLIP-2 paper)
        prompts = [f"Question: {q} Short answer:" for q in questions]
        # prompts = [self._build_prompt(q) for q in questions]
        
        # Process inputs
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Convert to appropriate dtype
        if self.precision == 'fp16':
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        elif self.precision == 'bf16':
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        
        # Generate answers with parameters from BLIP-2 paper
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=1,
                num_beams=self.num_beams,  # Paper uses 5
                length_penalty=self.length_penalty,  # Paper uses -1.0 to encourage shorter answers
                early_stopping=True,
                do_sample=False,  # Deterministic generation
                repetition_penalty=1.0,
                no_repeat_ngram_size=0
            )
        
        # Decode answers
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # Clean up answers - multiple strategies
        answers = []
        for text, prompt in zip(generated_texts, prompts):
            text = text.strip()
            
            # Remove prompt if it was repeated
            prompt_without_answer = prompt.replace(" Short answer:", "").strip()
            if text.lower().startswith(prompt_without_answer.lower()):
                text = text[len(prompt_without_answer):].strip()
            
            # Extract after "Short answer:" or "answer:" marker
            if "short answer:" in text.lower():
                parts = text.lower().split("short answer:")
                if len(parts) > 1:
                    text = text.split(":")[-1].strip()
            elif "answer:" in text.lower():
                parts = text.lower().split("answer:")
                if len(parts) > 1:
                    text = text.split(":")[-1].strip()
            
            # Remove "Question:" prefix if present
            if text.lower().startswith("question:"):
                text = text.split(":", 1)[1].strip() if ":" in text else text
            
            
            answers.append(text)
        
        return answers
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> VQAMetrics:
        """
        Evaluate BLIP2 on VQA dataset
        
        Args:
            dataloader: DataLoader for VQA dataset
            
        Returns:
            VQAMetrics object with evaluation results
        """
        all_predictions = []
        all_ground_truths = []
        
        print("Running VQA inference...")
        start_time = time.time()
        
        sample_count = 0
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            images = batch_data['image']
            questions = batch_data['question']
            answers_list = batch_data['answers']
            
            # Generate predictions
            predictions = self.answer_batch(images, questions)
            
            # Store results
            for pred, gt_answers in zip(
                predictions, answers_list
            ):
                all_predictions.append(pred)
                all_ground_truths.append(gt_answers)
        
        elapsed_time = time.time() - start_time
        print(f"Inference time: {elapsed_time:.2f}s")
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_predictions,
            all_ground_truths
        )
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: List[str],
        ground_truths: List[List[str]]
    ) -> VQAMetrics:
        """
        Calculate VQA evaluation metrics
        
        VQA accuracy follows the official VQA metric:
        min(# humans that said answer / 3, 1)
        """
        total_samples = len(predictions)
        
        # VQA Accuracy (official metric)
        vqa_scores = []
        exact_matches = []
        
        for pred, gt_list in zip(predictions, ground_truths):
            pred_clean = self._normalize_answer(pred)
            gt_clean = [self._normalize_answer(gt) for gt in gt_list]
            
            # VQA accuracy: min(# matching answers / 3, 1)
            matches = sum(1 for gt in gt_clean if pred_clean == gt)
            vqa_score = min(matches / 3.0, 1.0)
            vqa_scores.append(vqa_score)
            
            # Exact match (at least one answer matches)
            exact_match = 1.0 if any(pred_clean == gt for gt in gt_clean) else 0.0
            exact_matches.append(exact_match)
        
        vqa_accuracy = np.mean(vqa_scores)
        exact_match_accuracy = np.mean(exact_matches)
        
        
        return VQAMetrics(
            vqa_accuracy=vqa_accuracy,
            exact_match=exact_match_accuracy,
            total_samples=total_samples
        )
    
    def _calculate_breakdown_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[List[str]],
        categories: List[str]
    ) -> Dict[str, float]:
        """Calculate accuracy breakdown by category"""
        category_scores = defaultdict(list)
        
        for pred, gt_list, category in zip(predictions, ground_truths, categories):
            pred_clean = self._normalize_answer(pred)
            gt_clean = [self._normalize_answer(gt) for gt in gt_list]
            
            # VQA score
            matches = sum(1 for gt in gt_clean if pred_clean == gt)
            vqa_score = min(matches / 3.0, 1.0)
            
            category_scores[category].append(vqa_score)
        
        # Calculate mean for each category
        return {
            category: np.mean(scores)
            for category, scores in category_scores.items()
        }
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer text following VQA evaluation protocol
        """
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        return answer.strip()


def collate_fn(batch):
    """Custom collate function for VQA batches"""
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answers'] for item in batch]
    question_ids = [item['question_id'] for item in batch]
    
    return {
        'image': images,
        'question': questions,
        'answers': answers,
        'question_id': question_ids
    }


def EvaluateBLIP2VQA(
    model_name: str,
    annotations_file: str,
    questions_file: str,
    image_dir: str,
    device: str = "cuda:2",
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 10,
    num_beams: int = 5,
    length_penalty: float = -1.0,
    precision: str = 'fp16'
) -> VQAMetrics:
    """
    Evaluate BLIP2-FlanT5 model on VQAv2 dataset
    
    Args:
        model_name: BLIP2 model name or path
        annotations_file: Path to VQAv2 annotations JSON
        questions_file: Path to VQAv2 questions JSON
        image_dir: Path to COCO val2014 images
        device: CUDA device
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        max_samples: Maximum samples to evaluate (for testing)
        max_new_tokens: Maximum tokens to generate per answer (paper uses 10)
        num_beams: Beam search width (paper uses 5)
        length_penalty: Length penalty for generation (paper uses -1.0)
        precision: Model precision ('fp16', 'bf16', or 'fp32')
    
    Returns:
        VQAMetrics object with evaluation results
    """
    
    # Initialize model
    vqa_model = BLIP2VQA(
        model_name=model_name,
        device=device,
        precision=precision,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty
    )
    
    # Load dataset
    dataset = VQAv2Dataset(
        annotations_file=annotations_file,
        questions_file=questions_file,
        image_dir=image_dir,
        max_samples=max_samples
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Evaluate
    metrics = vqa_model.evaluate(dataloader)
    
    return metrics


if __name__ == "__main__":    
    metrics_t5 = EvaluateBLIP2VQA(
        model_name="models--Salesforce--blip2-flan-t5-xl/snapshots/0eb0d3b46c14c1f8c7680bca2693baafdb90bb28",
        annotations_file="Datasets/VQA-v2/v2_mscoco_val2014_annotations.json",
        questions_file="Datasets/VQA-v2/v2_OpenEnded_mscoco_val2014_questions.json",
        image_dir="Datasets/VQA-v2/val2014-typo",
        device='cuda:0',
        batch_size=32,
        max_new_tokens=10,
        num_beams=5,
        length_penalty=-1.0,
        precision='fp16'
    )
    
    print(f"VQA Accuracy: {metrics_t5.vqa_accuracy:.6f}")
    print(f"Exact Match: {metrics_t5.exact_match:.6f}")
    print(f"Total Samples: {metrics_t5.total_samples}")
