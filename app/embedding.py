from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingModel:
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
    
    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


embedding_model = EmbeddingModel()
