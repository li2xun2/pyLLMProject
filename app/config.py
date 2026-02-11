from pydantic_settings import BaseSettings
from typing import List, Dict


class Settings(BaseSettings):
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "mall"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "123456"
    DB_SCHEMA: str = "mall"
    
    EMBEDDING_MODEL: str = "C:/Users/lix/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
    EMBEDDING_DEVICE: str = "cuda"
    TOP_K: int = 3
    CONFIDENCE_THRESHOLD: float = 0.5
    
    LLM_MODEL: str = "C:/Users/lix/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    LLM_DEVICE: str = "cuda"
    LLM_MAX_LENGTH: int = 512
    LLM_TEMPERATURE: float = 0.7
    USE_LLM: bool = True
    
    VECTOR_TABLES: List[str] = ['ums_member', 'oms_order', 'ums_member_address', 'pms_product']
    VECTOR_TABLE_CONFIG: Dict[str, List[str]] = {
        'ums_member': ['nickname', 'phone_hidden', 'city', 'province'],
        'oms_order': ['order_sn', 'status', 'receiver_name', 'total_amount'],
        'ums_member_address': ['name', 'phone_hidden', 'province', 'city', 'district', 'detail_address'],
        'pms_product': ['name', 'brand_name', 'product_category_name', 'detail_html']
    }
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


settings = Settings()
