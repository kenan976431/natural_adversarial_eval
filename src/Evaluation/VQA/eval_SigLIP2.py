import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from typing import List, Optional
from dataclasses import dataclass
import re

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


class SigLIP2GemmaVQA:
    """SigLIP2 + Gemma2 VQA Model (PaliGemma2 architecture)"""
    
    def __init__(
        self,
        model_name: str,
        device: str = 'cuda:1',
        precision: str = 'fp16',
        max_new_tokens: int = 10,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0
    ):
        """
        Initialize SigLIP2 + Gemma2 VQA model (PaliGemma2 architecture)
        
        According to paper Section 3:
        - VQA tasks use greedy decoding (do_sample=False)
        - Short answers (max_new_tokens typically 10 for VQA)
        - No beam search for VQA (unlike captioning tasks)
        
        Architecture:
        - Frozen SigLIP2 visual encoder extracts image features
        - Linear projection layer maps visual features to text embedding space
        - Gemma2 language model generates answer autoregressively
        
        Args:
            model_name: PaliGemma2 model name or path
            device: CUDA device
            precision: Model precision ('fp16', 'bf16', or 'fp32')
            max_new_tokens: Maximum tokens to generate (paper uses ~10 for VQA)
            do_sample: Whether to sample (False for greedy decoding)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.device = device
        self.precision = precision
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        
        print(f"Loading SigLIP2 + Gemma2 model (PaliGemma2): {model_name}")
        print(f"Using greedy decoding for VQA (following paper)")
        
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
        
        # Load model with PaliGemma2 architecture
        # Contains: SigLIP2 encoder (frozen) + Linear projector + Gemma2 decoder
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device
        )
        self.model.eval()
        
        print(f"Model loaded successfully on {device}")
        print(f"Architecture: SigLIP2 (frozen visual encoder) + Gemma2 (language model)")
        print(f"Generation parameters: max_new_tokens={max_new_tokens}, greedy decoding")
    
    def _build_prompt(self, question: str) -> str:
        """
        Build prompt for PaliGemma2 VQA
        
        According to paper and Hugging Face documentation:
        - Format: "answer en {question}"
        - This is the standard VQA prompt format used in PaliGemma 2
        
        Note: The paper uses task-specific prefixes like:
        - "answer en" for English VQA
        - "caption en" for captioning
        - etc.
        """
        return f"answer en {question}"
    
    def answer_batch(
        self,
        images: List[Image.Image],
        questions: List[str]
    ) -> List[str]:
        """
        Generate answers for a batch of image-question pairs
        
        Args:
            images: List of PIL Images
            questions: List of questions
            
        Returns:
            List of predicted answers
        """
        
        # Format prompts with proper VQA prefix
        prompts = [self._build_prompt(q) for q in questions]
        
        # Process inputs
        # PaliGemma2 processor handles both image and text
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=True
        ).to(self.device)
        
        # Convert to appropriate dtype
        if self.precision == 'fp16':
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        elif self.precision == 'bf16':
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        
        # Generate answers using greedy decoding (following paper)
        # Paper Section 3: VQA uses greedy decoding, not beam search
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode answers, skipping input prompt tokens
        input_len = inputs["input_ids"].shape[-1]
        generated_texts = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True
        )
        
        # Clean up answers - minimal processing
        answers = []
        for text in generated_texts:
            # Remove leading/trailing whitespace
            cleaned = text.strip()
            
            # Split on newline (model might generate multiple lines)
            if '\n' in cleaned:
                cleaned = cleaned.split('\n')[0].strip()
            
            # Handle empty predictions
            if not cleaned:
                cleaned = ""
            
            answers.append(cleaned)
        
        return answers
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> VQAMetrics:
        """
        Evaluate SigLIP2+Gemma2 model on VQA dataset
        
        Args:
            dataloader: DataLoader for VQA dataset
            
        Returns:
            VQAMetrics object with evaluation results
        """
        all_predictions = []
        all_ground_truths = []
        
        print("Running VQA inference with SigLIP2 + Gemma2...")
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch_data['image']
            questions = batch_data['question']
            answers_list = batch_data['answers']
            
            predictions = self.answer_batch(images, questions)
            
            # Store results
            for pred, gt_answers in zip(predictions, answers_list):
                all_predictions.append(pred)
                all_ground_truths.append(gt_answers)
        
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
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer text following VQA evaluation protocol
        
        This follows the standard VQA normalization:
        1. Convert to lowercase
        2. Remove punctuation
        3. Remove articles (a, an, the)
        4. Remove extra whitespace
        """
        if not answer:
            return ""
        
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


def EvaluateSigLIP2GemmaVQA(
    model_name: str,
    annotations_file: str,
    questions_file: str,
    image_dir: str,
    device: str = "cuda:1",
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 10,
    precision: str = 'bf16',
    do_sample: bool = False
) -> VQAMetrics:
    """
    Evaluate SigLIP2 + Gemma2 model (PaliGemma2) on VQAv2 dataset
    
    Architecture:
    - SigLIP2 vision encoder (frozen): Extracts visual features from images
    - Linear projection: Maps visual tokens to text embedding dimensions
    - Gemma2 language model: Generates answers autoregressively
    
    Training details from paper (Section 3):
    - Stage 3 fine-tuning on VQA uses:
      * Learning rate sweep: {0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0} × 10^-5
      * Greedy decoding (not beam search)
      * Short sequence length for VQA (~10 tokens)
    
    Args:
        model_name: PaliGemma2 model name or path
                   - Pre-trained: 'google/paligemma2-3b-pt-448' (need fine-tuning)
                   - Fine-tuned: Use VQA fine-tuned checkpoints if available
        annotations_file: Path to VQAv2 annotations JSON
        questions_file: Path to VQAv2 questions JSON
        image_dir: Path to COCO val2014 images
        device: CUDA device
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        max_samples: Maximum samples to evaluate (for testing)
        max_new_tokens: Maximum tokens to generate (10 for VQA)
        precision: Model precision ('bf16' recommended by paper)
        do_sample: Whether to sample (False for greedy decoding)
    
    Returns:
        VQAMetrics object with evaluation results
        
    Note:
        According to paper Table 13, PaliGemma 2 achieves:
        - 3B @ 224px²: 83.0% VQA accuracy
        - 3B @ 448px²: 84.8% VQA accuracy
        - 10B @ 448px²: 85.8% VQA accuracy
    """
    
    # Initialize model
    vqa_model = SigLIP2GemmaVQA(
        model_name=model_name,
        device=device,
        precision=precision,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.0,
        top_p=1.0
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
    metrics = EvaluateSigLIP2GemmaVQA(
        model_name="models--google--paligemma2-3b-pt-448/snapshots/bd60523c1ff2b0020c11e68affe65ef0b2379cab",
        annotations_file="Datasets/VQA-v2/v2_mscoco_val2014_annotations.json",
        questions_file="Datasets/VQA-v2/v2_OpenEnded_mscoco_val2014_questions.json",
        image_dir="Datasets/VQA-v2/val2014-typo",
        device='cuda:0',
        batch_size=32,
        max_new_tokens=10,
        precision='bf16',
        do_sample=False
    )

    print(f"VQA Accuracy: {metrics.vqa_accuracy:.6f}")
    print(f"Exact Match: {metrics.exact_match:.6f}")
    print(f"Total Samples: {metrics.total_samples}")
