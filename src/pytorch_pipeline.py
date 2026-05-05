import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Tuple

class SemanticMatcher:
    """
    PyTorch-based semantic matching using Sentence Transformers.
    Compares resume and job description at the semantic level,
    beyond keyword matching.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Loaded {model_name} on {self.device}")
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode list of texts to embeddings."""
        embeddings = self.model.encode(
            texts, 
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device
        )
        return embeddings
    
    def compute_similarity(self, resume: str, job_description: str) -> float:
        """Compute cosine similarity between resume and JD."""
        embeddings = self.encode_texts([resume, job_description])
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return similarity.item()
    
    def rank_candidates(
        self, 
        resumes: List[str], 
        job_description: str
    ) -> List[Tuple[int, float]]:
        """
        Rank multiple resumes against a single job description.
        Returns list of (index, score) sorted by score descending.
        """
        # Encode all at once (batch processing)
        jd_embedding = self.model.encode(job_description, convert_to_tensor=True)
        resume_embeddings = self.model.encode(resumes, convert_to_tensor=True)
        
        # Compute similarities
        similarities = util.cos_sim(jd_embedding, resume_embeddings)[0]
        
        # Sort by score
        ranked = sorted(
            enumerate(similarities.tolist()), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked
    
    def compare_ml_vs_dl(
        self, 
        sklearn_scores: List[float], 
        semantic_scores: List[float],
        labels: List[int]
    ) -> dict:
        """
        Compare scikit-learn (ML) vs Sentence Transformers (DL) predictions.
        This is the KEY comparison that shows depth of understanding.
        """
        from sklearn.metrics import f1_score, accuracy_score
        
        # Threshold semantic scores to binary
        semantic_predictions = [1 if s > 0.5 else 0 for s in semantic_scores]
        sklearn_predictions = [1 if s > 0.5 else 0 for s in sklearn_scores]
        
        return {
            'sklearn_f1': f1_score(labels, sklearn_predictions, average='weighted'),
            'semantic_f1': f1_score(labels, semantic_predictions, average='weighted'),
            'sklearn_accuracy': accuracy_score(labels, sklearn_predictions),
            'semantic_accuracy': accuracy_score(labels, semantic_predictions),
        }
