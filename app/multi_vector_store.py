import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from app.embedding import embedding_model
from app.config import settings


class MultiTableVectorStore:
    def __init__(self):
        self.indices = {}
        self.data = {}
        self.dimension = embedding_model.dimension
        self._initialized_tables = set()
        self._initialize_indices()
    
    def _initialize_indices(self):
        pass
    
    def add_table_data(self, table_name: str, data: List[Dict], text_columns: List[str]):
        if not data:
            return
        
        if table_name in self._initialized_tables:
            print(f"Table '{table_name}' already initialized, skipping...")
            return
        
        self.data[table_name] = data
        
        texts = []
        for row in data:
            combined_text = ' '.join([str(row.get(col, '')) for col in text_columns if row.get(col)])
            texts.append(combined_text)
        
        if not texts:
            return
        
        embeddings = embedding_model.encode(texts)
        
        index = faiss.IndexFlatIP(self.dimension)
        index.reset()
        index.add(embeddings.astype('float32'))
        
        self.indices[table_name] = {
            'index': index,
            'data': data,
            'text_columns': text_columns
        }
        
        self._initialized_tables.add(table_name)
        print(f"Added {len(data)} records from table '{table_name}' to vector store")
    
    def search(self, query: str, tables: List[str] = None, top_k: int = None) -> List[Tuple[Dict, float, str]]:
        if not self.indices:
            return []
        
        top_k = top_k or settings.TOP_K
        
        if tables:
            target_indices = {k: v for k, v in self.indices.items() if k in tables}
        else:
            target_indices = self.indices
        
        if not target_indices:
            return []
        
        query_embedding = embedding_model.encode_single(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        all_results = []
        
        for table_name, index_info in target_indices.items():
            index = index_info['index']
            data = index_info['data']
            
            if len(data) == 0:
                continue
            
            scores, indices = index.search(query_embedding, min(top_k, len(data)))
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(data):
                    all_results.append((data[idx], float(score), table_name))
        
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return all_results[:top_k]
    
    def get_best_match(self, query: str, tables: List[str] = None) -> Tuple[Optional[Dict], float, Optional[str]]:
        results = self.search(query, tables=tables, top_k=1)
        if results:
            return results[0]
        return None, 0.0, None
    
    def get_table_stats(self, table_name: str) -> Dict:
        if table_name not in self.indices:
            return {'size': 0, 'columns': []}
        
        index_info = self.indices[table_name]
        return {
            'size': index_info['index'].ntotal,
            'columns': index_info['text_columns']
        }
    
    def get_all_stats(self) -> Dict:
        stats = {}
        for table_name, index_info in self.indices.items():
            stats[table_name] = {
                'size': index_info['index'].ntotal,
                'columns': index_info['text_columns']
            }
        return stats
    
    def get_total_size(self) -> int:
        total = 0
        for index_info in self.indices.values():
            total += index_info['index'].ntotal
        return total


multi_vector_store = MultiTableVectorStore()
