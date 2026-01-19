from pydantic_settings import BaseSettings
from typing import List, Dict


class Settings(BaseSettings):
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "shop"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "123456"
    
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DEVICE: str = "cuda"
    TOP_K: int = 3
    CONFIDENCE_THRESHOLD: float = 0.5
    
    LLM_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
    LLM_DEVICE: str = "cuda"
    LLM_MAX_LENGTH: int = 512
    LLM_TEMPERATURE: float = 0.7
    USE_LLM: bool = True
    
    VECTOR_TABLES: List[str] = ['faq', 'products', 'orders', 'customers']
    VECTOR_TABLE_CONFIG: Dict[str, List[str]] = {
        'faq': ['question', 'answer'],
        'products': ['name', 'description', 'category'],
        'orders': ['order_id', 'status', 'customer_name'],
        'customers': ['name', 'email', 'phone', 'address']
    }
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


settings = Settings()
