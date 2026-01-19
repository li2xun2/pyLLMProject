import numpy as np
import faiss
from typing import List, Dict, Tuple
from app.embedding import embedding_model
from app.config import settings


class VectorStore:
    def __init__(self):
        self.index = None
        self.faq_data = []
        self.dimension = embedding_model.dimension
        self._initialize_index()
    
    def _initialize_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
    
    def add_faqs(self, faqs: List[Dict]):
        if not faqs:
            return
        
        self.faq_data = faqs
        questions = [faq['question'] for faq in faqs]
        embeddings = embedding_model.encode(questions)
        
        self.index.reset()
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        if not self.faq_data:
            return []
        
        top_k = top_k or settings.TOP_K
        query_embedding = embedding_model.encode_single(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.faq_data)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.faq_data):
                results.append((self.faq_data[idx], float(score)))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_best_match(self, query: str) -> Tuple[Dict, float]:
        results = self.search(query, top_k=1)
        if results:
            return results[0]
        return None, 0.0
    
    def get_size(self) -> int:
        return self.index.ntotal if self.index else 0


vector_store = VectorStore()
