import psycopg2
from psycopg2.extras import DictCursor
from typing import List, Dict, Optional
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.connection = None
        self._create_config_table()
    
    def connect(self):
        try:
            logger.debug(f"Connecting to database: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
            connection = psycopg2.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                dbname=settings.DB_NAME,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                options="-c client_encoding=utf8"
            )
            logger.debug("Database connection successful")
            return connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _create_config_table(self):
        """创建AI配置表"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {settings.DB_SCHEMA}.ai_config (
                        id SERIAL PRIMARY KEY,
                        config_key VARCHAR(255) UNIQUE NOT NULL,
                        config_value TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.info("AI config table created or already exists")
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error creating config table: {e}")
    
    def disconnect(self):
        if self.connection and not self.connection.closed:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
    
    def test_connection(self) -> bool:
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_all_tables(self) -> List[str]:
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            try:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = %s
                    AND table_name NOT IN ('spatial_ref_sys', 'geography_columns', 'geometry_columns')
                    ORDER BY table_name
                """, (settings.DB_SCHEMA,))
                tables = [row['table_name'] for row in cursor.fetchall()]
                logger.info(f"Found tables: {tables}")
                return tables
            finally:
                cursor.close()
        except Exception as e:
            logger.error(f"Error getting tables: {e}")
            return []
    
    def get_table_columns(self, table_name: str) -> List[Dict]:
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            try:
                cursor.execute("""
                    SELECT column_name, data_type, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_name = %s
                    AND table_schema = %s
                    ORDER BY ordinal_position
                """, (table_name, settings.DB_SCHEMA))
                columns = cursor.fetchall()
                logger.info(f"Table {table_name} columns: {[col['column_name'] for col in columns]}")
                return columns
            finally:
                cursor.close()
        except Exception as e:
            logger.error(f"Error getting table columns: {e}")
            return []
    
    def fetch_table_data(self, table_name: str, text_columns: List[str], limit: int = None, user_id: Optional[str] = None) -> List[Dict]:
        conn = None
        cursor = None
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            columns_str = ', '.join(text_columns)
            # 添加schema前缀
            query = f"SELECT {columns_str} FROM {settings.DB_SCHEMA}.{table_name}"
            params = []
            
            # 添加用户ID过滤
            if user_id:
                # 根据表名使用不同的用户ID字段
                user_id_fields = {
                    'oms_order': 'member_id',
                    'ums_member': 'id',
                    'ums_member_address': 'member_id',
                    'ums_member_cart': 'member_id'
                }
                
                if table_name in user_id_fields:
                    user_field = user_id_fields[table_name]
                    query += f" WHERE {user_field} = %s"
                    params.append(user_id)
                    logger.info(f"Adding user filter: {user_field} = {user_id} to table {table_name}")
            
            if limit:
                query += f" LIMIT {limit}"
            
            logger.info(f"Executing query: {query}")
            logger.info(f"Query params: {params}")
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            results = cursor.fetchall()
            
            processed_results = []
            for row in results:
                processed_row = {}
                for col in text_columns:
                    if col in row and row[col]:
                        processed_row[col] = str(row[col])
                if processed_row:
                    processed_row['_table'] = table_name
                    if user_id:
                        processed_row['_user_id'] = user_id
                    processed_results.append(processed_row)
            
            logger.info(f"Fetched {len(processed_results)} rows from {table_name}")
            return processed_results
        except Exception as e:
            logger.error(f"Error fetching table data: {e}")
            return []
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def fetch_table_data_since(self, table_name: str, text_columns: List[str], since_time: str, user_id: Optional[str] = None) -> List[Dict]:
        """增量获取表数据，获取指定时间之后的数据"""
        conn = None
        cursor = None
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            # 检查表是否有时间戳字段
            timestamp_fields = {
                'oms_order': 'create_time',
                'ums_member': 'create_time',
                'ums_member_address': 'create_time',
                'pms_product': 'create_time'
            }
            
            if table_name not in timestamp_fields:
                logger.warning(f"Table {table_name} does not have a timestamp field, skipping incremental update")
                return []
            
            timestamp_field = timestamp_fields[table_name]
            columns_str = f"{', '.join(text_columns)}, {timestamp_field}"
            
            # 添加schema前缀
            query = f"SELECT {columns_str} FROM {settings.DB_SCHEMA}.{table_name} WHERE {timestamp_field} > %s"
            params = [since_time]
            
            # 添加用户ID过滤
            if user_id:
                # 根据表名使用不同的用户ID字段
                user_id_fields = {
                    'oms_order': 'member_id',
                    'ums_member': 'id',
                    'ums_member_address': 'member_id',
                    'ums_member_cart': 'member_id'
                }
                
                if table_name in user_id_fields:
                    user_field = user_id_fields[table_name]
                    query += f" AND {user_field} = %s"
                    params.append(user_id)
                    logger.info(f"Adding user filter: {user_field} = {user_id} to table {table_name}")
            
            logger.info(f"Executing incremental query: {query}")
            logger.info(f"Query params: {params}")
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            processed_results = []
            for row in results:
                processed_row = {}
                for col in text_columns:
                    if col in row and row[col]:
                        processed_row[col] = str(row[col])
                if processed_row:
                    processed_row['_table'] = table_name
                    processed_row['_timestamp'] = str(row[timestamp_field])
                    if user_id:
                        processed_row['_user_id'] = user_id
                    processed_results.append(processed_row)
            
            logger.info(f"Fetched {len(processed_results)} new rows from {table_name} since {since_time}")
            return processed_results
        except Exception as e:
            logger.error(f"Error fetching incremental table data: {e}")
            return []
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def fetch_user_orders(self, user_id: str, limit: int = 50) -> List[Dict]:
        """获取用户订单信息"""
        conn = None
        cursor = None
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            query = f"""
                SELECT id, order_sn, status, create_time, total_amount 
                FROM {settings.DB_SCHEMA}.oms_order 
                WHERE member_id = %s 
                ORDER BY create_time DESC 
                LIMIT %s
            """
            cursor.execute(query, (user_id, limit))
            results = cursor.fetchall()
            logger.info(f"Fetched {len(results)} orders for user {user_id}")
            return results
        except Exception as e:
            logger.error(f"Error fetching user orders: {e}")
            return []
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def fetch_user_addresses(self, user_id: str) -> List[Dict]:
        """获取用户地址信息"""
        conn = None
        cursor = None
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            query = f"""
                SELECT id, name, phone_hidden, province, city, district, detail_address, is_default 
                FROM {settings.DB_SCHEMA}.ums_member_address 
                WHERE member_id = %s 
                ORDER BY is_default DESC, id DESC
            """
            cursor.execute(query, (user_id,))
            results = cursor.fetchall()
            logger.info(f"Fetched {len(results)} addresses for user {user_id}")
            return results
        except Exception as e:
            logger.error(f"Error fetching user addresses: {e}")
            return []
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def fetch_user_profile(self, user_id: str) -> Optional[Dict]:
        """获取用户个人信息"""
        conn = None
        cursor = None
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            query = f"""
                SELECT id, nickname, phone_hidden as phone, avatar, city, province 
                FROM {settings.DB_SCHEMA}.ums_member 
                WHERE id = %s
            """
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
            if result:
                logger.info(f"Fetched profile for user {user_id}")
            else:
                logger.warning(f"No profile found for user {user_id}")
            return result
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def fetch_all_faqs(self) -> List[Dict]:
        try:
            # 数据库中没有faq表，返回空结果
            logger.info("No FAQ table found in database, returning empty results")
            return []
        except Exception as e:
            logger.error(f"Error fetching FAQs: {e}")
            return []
    
    def fetch_faq_by_id(self, faq_id: int) -> Optional[Dict]:
        try:
            # 数据库中没有faq表，返回空结果
            logger.info("No FAQ table found in database, returning empty results")
            return None
        except Exception as e:
            logger.error(f"Error fetching FAQ by id: {e}")
            return None
    
    def get_ai_config(self, config_key: str) -> Optional[str]:
        """从数据库获取AI配置"""
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            try:
                cursor.execute(f"""
                    SELECT config_value 
                    FROM {settings.DB_SCHEMA}.ai_config 
                    WHERE config_key = %s
                """, (config_key,))
                result = cursor.fetchone()
                if result:
                    logger.info(f"Fetched AI config for key: {config_key}")
                    return result['config_value']
                logger.info(f"No AI config found for key: {config_key}")
                return None
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error getting AI config: {e}")
            return None
    
    def save_ai_config(self, config_key: str, config_value: str) -> bool:
        """保存AI配置到数据库"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    INSERT INTO {settings.DB_SCHEMA}.ai_config 
                    (config_key, config_value, updated_at) 
                    VALUES (%s, %s, CURRENT_TIMESTAMP) 
                    ON CONFLICT (config_key) 
                    DO UPDATE SET 
                        config_value = %s, 
                        updated_at = CURRENT_TIMESTAMP
                """, (config_key, config_value, config_value))
                conn.commit()
                logger.info(f"Saved AI config for key: {config_key}")
                return True
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error saving AI config: {e}")
            return False


db = Database()
