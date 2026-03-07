from pydantic_settings import BaseSettings
from typing import List, Dict
import json


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
    
    # 默认值
    _default_vector_tables: List[str] = ['ums_member', 'oms_order', 'ums_member_address', 'pms_product', 'pms_sku']
    _default_vector_table_config: Dict[str, List[str]] = {
        'ums_member': ['nickname', 'phone_hidden', 'city', 'province'],
        'oms_order': ['order_sn', 'status', 'receiver_name', 'total_amount'],
        'ums_member_address': ['name', 'phone_hidden', 'province', 'city', 'district', 'detail_address'],
        'pms_product': ['name', 'brand_name', 'product_category_name', 'detail_html', 'price', 'unit'],

        'pms_sku': ['out_sku_id', 'price', 'pic', 'stock', 'sp_data']
    }
    _default_vector_table_keywords: Dict[str, List[str]] = {
        'ums_member': ['用户', '会员', '个人信息', '账号', '登录', '注册', '个人资料', '我的信息'],
        'oms_order': ['订单', '购买', '交易', '支付', '物流', '发货', '收货', '订单状态', '订单详情'],
        'ums_member_address': ['地址', '收货地址', '我的地址', '地址管理', '收货信息'],
        'pms_product': ['商品', '产品', '货物', '库存', '价格', '品牌', '类别', '商品详情', '产品信息'],
        'pms_sku': ['库存', 'SKU', '商品库存', '库存信息', '库存状态', '库存数量']
    }
    
    # 实际使用的值，会从数据库加载
    VECTOR_TABLES: List[str] = _default_vector_tables
    VECTOR_TABLE_CONFIG: Dict[str, List[str]] = _default_vector_table_config
    VECTOR_TABLE_KEYWORDS: Dict[str, List[str]] = _default_vector_table_keywords
    
    def load_from_db(self, db):
        """从数据库加载配置"""
        try:
            # 尝试从数据库加载配置
            vector_tables_str = db.get_ai_config("vector_tables")
            vector_table_config_str = db.get_ai_config("vector_table_config")
            vector_table_keywords_str = db.get_ai_config("vector_table_keywords")
            
            if vector_tables_str:
                self.VECTOR_TABLES = json.loads(vector_tables_str)
            if vector_table_config_str:
                try:
                    config = json.loads(vector_table_config_str)
                    # 检查配置是否包含null值
                    has_null = False
                    for table, columns in config.items():
                        if any(col is None for col in columns):
                            has_null = True
                            break
                    if has_null:
                        # 如果包含null值，使用默认配置
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning("Database config contains null values, using default configuration")
                        self.VECTOR_TABLE_CONFIG = self._default_vector_table_config
                    else:
                        self.VECTOR_TABLE_CONFIG = config
                except:
                    # 如果解析失败，使用默认配置
                    self.VECTOR_TABLE_CONFIG = self._default_vector_table_config
            if vector_table_keywords_str:
                self.VECTOR_TABLE_KEYWORDS = json.loads(vector_table_keywords_str)
        except Exception as e:
            # 如果加载失败，使用默认值
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error loading config from database: {e}")
            logger.info("Using default configuration")
            self.VECTOR_TABLES = self._default_vector_tables
            self.VECTOR_TABLE_CONFIG = self._default_vector_table_config
            self.VECTOR_TABLE_KEYWORDS = self._default_vector_table_keywords
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


settings = Settings()
