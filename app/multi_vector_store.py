import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from app.embedding import embedding_model
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTableVectorStore:
    def __init__(self):
        self.indices = {}
        self.data = {}
        self.dimension = embedding_model.dimension
        self._initialized_tables = set()
        self._initialize_indices()
        logger.info(f"MultiTableVectorStore initialized with dimension: {self.dimension}")
    
    def _initialize_indices(self):
        pass
    
    def add_table_data(self, table_name: str, data: List[Dict], text_columns: List[str]):
        if not data:
            logger.warning(f"No data provided for table '{table_name}'")
            return
        
        if table_name in self._initialized_tables:
            logger.info(f"Table '{table_name}' already initialized, skipping...")
            return
        
        logger.info(f"Processing {len(data)} records from table '{table_name}'")
        self.data[table_name] = data
        
        texts = []
        valid_records = []
        
        for row in data:
            # 智能合并文本，添加字段名以提供更多上下文
            text_parts = []
            for col in text_columns:
                value = row.get(col)
                if value:
                    # 添加字段名作为前缀，增强语义信息
                    text_parts.append(f"{col}: {str(value)}")
            
            combined_text = ' '.join(text_parts)
            if combined_text.strip():
                texts.append(combined_text)
                valid_records.append(row)
        
        if not texts:
            logger.warning(f"No valid text data found in table '{table_name}'")
            return
        
        logger.info(f"Generating embeddings for {len(texts)} valid records")
        try:
            embeddings = embedding_model.encode(texts)
            logger.info(f"Embeddings generated successfully: shape={embeddings.shape}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return
        
        # 使用更高效的索引结构
        try:
            index = faiss.IndexFlatIP(self.dimension)
            index.reset()
            index.add(embeddings.astype('float32'))
            logger.info(f"Added {index.ntotal} embeddings to index for table '{table_name}'")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return
        
        self.indices[table_name] = {
            'index': index,
            'data': valid_records,
            'text_columns': text_columns
        }
        
        self._initialized_tables.add(table_name)
        logger.info(f"Successfully added {len(valid_records)} records from table '{table_name}' to vector store")
    
    def update_table_data(self, table_name: str, data: List[Dict], text_columns: List[str]):
        """增量更新表数据"""
        if table_name in self._initialized_tables:
            # 移除旧数据
            del self.indices[table_name]
            if table_name in self.data:
                del self.data[table_name]
            self._initialized_tables.remove(table_name)
            logger.info(f"Removed existing data for table '{table_name}' for update")
        
        # 添加新数据
        self.add_table_data(table_name, data, text_columns)
    
    def add_new_data(self, table_name: str, new_data: List[Dict], text_columns: List[str]):
        """向现有索引添加新数据"""
        if table_name not in self._initialized_tables:
            logger.warning(f"Table '{table_name}' not initialized, use add_table_data instead")
            return
        
        if not new_data:
            logger.warning(f"No new data provided for table '{table_name}'")
            return
        
        logger.info(f"Processing {len(new_data)} new records for table '{table_name}'")
        
        texts = []
        valid_records = []
        
        for row in new_data:
            # 智能合并文本，添加字段名以提供更多上下文
            text_parts = []
            for col in text_columns:
                value = row.get(col)
                if value:
                    # 添加字段名作为前缀，增强语义信息
                    text_parts.append(f"{col}: {str(value)}")
            
            combined_text = ' '.join(text_parts)
            if combined_text.strip():
                texts.append(combined_text)
                valid_records.append(row)
        
        if not texts:
            logger.warning(f"No valid text data found in new records for table '{table_name}'")
            return
        
        logger.info(f"Generating embeddings for {len(texts)} valid new records")
        try:
            embeddings = embedding_model.encode(texts)
            logger.info(f"Embeddings generated successfully: shape={embeddings.shape}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return
        
        # 获取现有索引并添加新数据
        try:
            index_info = self.indices[table_name]
            index = index_info['index']
            
            # 向现有索引添加新嵌入
            index.add(embeddings.astype('float32'))
            logger.info(f"Added {len(valid_records)} new embeddings to index for table '{table_name}'")
            
            # 更新数据存储
            index_info['data'].extend(valid_records)
            if table_name in self.data:
                self.data[table_name].extend(new_data)
            
            logger.info(f"Successfully added {len(valid_records)} new records to table '{table_name}'")
        except Exception as e:
            logger.error(f"Error adding new data to table '{table_name}': {e}")
            return
    
    def search(self, query: str, tables: List[str] = None, top_k: int = None) -> List[Tuple[Dict, float, str]]:
        if not self.indices:
            logger.warning("No indices available for search")
            return []
        
        top_k = top_k or settings.TOP_K
        logger.info(f"Searching for: '{query}' with top_k={top_k}")
        
        if tables:
            target_indices = {k: v for k, v in self.indices.items() if k in tables}
            logger.info(f"Searching in tables: {list(target_indices.keys())}")
        else:
            target_indices = self.indices
            logger.info(f"Searching in all {len(target_indices)} tables")
        
        if not target_indices:
            logger.warning("No matching tables found for search")
            return []
        
        try:
            query_embedding = embedding_model.encode_single(query)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            return []
        
        all_results = []
        
        for table_name, index_info in target_indices.items():
            index = index_info['index']
            data = index_info['data']
            
            if len(data) == 0:
                continue
            
            try:
                scores, indices = index.search(query_embedding, min(top_k, len(data)))
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(data):
                        all_results.append((data[idx], float(score), table_name))
                        logger.debug(f"Match found: score={float(score):.4f} in table '{table_name}'")
            except Exception as e:
                logger.error(f"Error searching in table '{table_name}': {e}")
                continue
        
        # 按相似度排序
        all_results.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Found {len(all_results)} total matches")
        
        return all_results[:top_k]
    
    def get_best_match(self, query: str, tables: List[str] = None) -> Tuple[Optional[Dict], float, Optional[str]]:
        results = self.search(query, tables=tables, top_k=1)
        if results:
            logger.info(f"Best match found with score: {results[0][1]:.4f}")
            return results[0]
        logger.warning("No match found for query")
        return None, 0.0, None
    
    def get_table_stats(self, table_name: str) -> Dict:
        if table_name not in self.indices:
            return {'size': 0, 'columns': []}
        
        index_info = self.indices[table_name]
        stats = {
            'size': index_info['index'].ntotal,
            'columns': index_info['text_columns']
        }
        logger.debug(f"Stats for table '{table_name}': {stats}")
        return stats
    
    def get_all_stats(self) -> Dict:
        stats = {}
        total_size = 0
        
        for table_name, index_info in self.indices.items():
            table_stats = {
                'size': index_info['index'].ntotal,
                'columns': index_info['text_columns']
            }
            stats[table_name] = table_stats
            total_size += table_stats['size']
        
        logger.info(f"Total vector store stats: {total_size} records across {len(stats)} tables")
        return stats
    
    def get_total_size(self) -> int:
        total = 0
        for index_info in self.indices.values():
            total += index_info['index'].ntotal
        return total
    
    def clear(self):
        """清空所有索引和数据"""
        self.indices.clear()
        self.data.clear()
        self._initialized_tables.clear()
        logger.info("MultiTableVectorStore cleared")


multi_vector_store = MultiTableVectorStore()
