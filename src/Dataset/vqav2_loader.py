"""
VQAv2 Dataset Loader
"""

import os
import json
from PIL import Image
from typing import Optional
from torch.utils.data import Dataset


class VQAv2Dataset(Dataset):
    """
    VQAv2 Dataset Loader
    Expects data structure:
    - annotations: JSON file with questions and answers
    - images: Directory containing image files
    """
    
    def __init__(
        self,
        annotations_file: str,
        questions_file: str,
        image_dir: str,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            annotations_file: Path to v2_mscoco_val2014_annotations.json
            questions_file: Path to v2_OpenEnded_mscoco_val2014_questions.json
            image_dir: Path to val2014 images directory
            max_samples: Maximum number of samples to load
        """
        self.image_dir = image_dir
        
        print(f"Loading VQAv2 annotations from {annotations_file}")
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)
        
        print(f"Loading VQAv2 questions from {questions_file}")
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        
        # Build question_id to question mapping
        self.questions = {q['question_id']: q for q in questions_data['questions']}
        
        # Build samples
        self.samples = []
        for ann in annotations_data['annotations']:
            question_id = ann['question_id']
            if question_id not in self.questions:
                continue
                
            question_info = self.questions[question_id]
            
            # Extract multiple answers
            answers = [a['answer'] for a in ann['answers']]
            
            self.samples.append({
                'question_id': question_id,
                'image_id': ann['image_id'],
                'question': question_info['question'],
                'answers': answers,
                'answer_type': ann['answer_type'],
                'question_type': ann['question_type'],
                'multiple_choice_answer': ann['multiple_choice_answer']
            })
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} VQA samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_filename = f"COCO_val2014_{sample['image_id']:012d}.jpg"
        image_path = os.path.join(self.image_dir, image_filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image in case of error
            image = Image.new('RGB', (224, 224), color='black')
        
        return {
            'image': image,
            'question': sample['question'],
            'answers': sample['answers'],
            'answer_type': sample['answer_type'],
            'question_type': sample['question_type'],
            'question_id': sample['question_id']
        }
