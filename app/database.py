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
            logger.info(f"Connecting to database: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
            logger.info(f"User: {settings.DB_USER}, Schema: {settings.DB_SCHEMA}")
            connection = psycopg2.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                dbname=settings.DB_NAME,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                options="-c client_encoding=utf8"
            )
            logger.info("Database connection successful")
            return connection
        except Exception as e:
            logger.error(f"Database connection failed: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
                # 将DictCursor对象转换为普通字典
                columns = [dict(col) for col in columns]
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
                    try:
                        if col in row and row[col] is not None:
                            processed_row[col] = str(row[col])
                    except Exception as e:
                        logger.error(f"Error processing column {col}: {e}")
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
                    try:
                        if col in row and row[col] is not None:
                            processed_row[col] = str(row[col])
                    except Exception as e:
                        logger.error(f"Error processing column {col}: {e}")
                if processed_row:
                    processed_row['_table'] = table_name
                    try:
                        processed_row['_timestamp'] = str(row[timestamp_field])
                    except Exception as e:
                        logger.error(f"Error processing timestamp field {timestamp_field}: {e}")
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
        """获取用户个人信息（同时检查普通用户和管理员）"""
        conn = None
        cursor = None
        try:
            # 检查用户ID是否有效
            if not user_id or user_id == 'anonymous':
                logger.warning(f"Invalid user ID: {user_id}")
                return None
            
            # 检查是否为管理员token（admin_前缀）
            if user_id.startswith('admin_'):
                logger.info(f"Admin token detected: {user_id}")
                # 返回一个默认的管理员信息
                return {
                    'id': user_id,
                    'nickname': '管理员',
                    'phone': None,
                    'avatar': None,
                    'city': None,
                    'province': None,
                    'user_type': 'admin'
                }
            
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            # 先检查普通用户表
            query = f"""
                SELECT id, nickname, phone_hidden as phone, avatar, city, province 
                FROM {settings.DB_SCHEMA}.ums_member 
            """
            
            cursor.execute(query)
            members = cursor.fetchall()
            
            # 尝试找到匹配的用户
            for member in members:
                # 检查ID是否匹配
                if str(member['id']) == user_id:
                    logger.info(f"Fetched profile for user {user_id} from ums_member by ID")
                    # 将DictCursor对象转换为普通字典
                    member_dict = dict(member)
                    member_dict['user_type'] = 'member'
                    return member_dict
            
            # 如果普通用户表中没有，检查管理员表
            query = f"""
                SELECT user_id as id, nick_name as nickname, phonenumber as phone 
                FROM {settings.DB_SCHEMA}.sys_user 
            """
            
            cursor.execute(query)
            admins = cursor.fetchall()
            
            # 尝试找到匹配的管理员
            for admin in admins:
                # 检查ID是否匹配
                if str(admin['id']) == user_id:
                    logger.info(f"Fetched profile for user {user_id} from sys_user by ID")
                    # 将DictCursor对象转换为普通字典
                    admin_dict = dict(admin)
                    admin_dict['user_type'] = 'admin'
                    return admin_dict
            
            # 如果都没有找到
            logger.warning(f"No profile found for user {user_id}")
            return None
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
    
    def fetch_all_faqs(self, page: int = 1, size: int = 10, keyword: str = None) -> List[Dict]:
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            try:
                # 构建查询
                query = f"""
                    SELECT id, question, answer, category, sort, status, create_time
                    FROM {settings.DB_SCHEMA}.faq
                    WHERE status = 1
                """
                params = []
                
                # 添加搜索条件
                if keyword:
                    query += " AND (question ILIKE %s OR answer ILIKE %s OR category ILIKE %s)"
                    params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
                
                # 添加排序
                query += " ORDER BY id DESC"
                
                # 添加分页
                offset = (page - 1) * size
                query += " LIMIT %s OFFSET %s"
                params.extend([size, offset])
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                # 将DictCursor对象转换为普通字典
                faqs = [dict(faq) for faq in results]
                logger.info(f"Fetched {len(faqs)} FAQs (page {page}, size {size})")
                return faqs
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error fetching FAQs: {e}")
            return []
    
    def count_faqs(self, keyword: str = None) -> int:
        try:
            conn = self.connect()
            cursor = conn.cursor()
            try:
                # 构建查询
                query = f"""
                    SELECT COUNT(*)
                    FROM {settings.DB_SCHEMA}.faq
                    WHERE status = 1
                """
                params = []
                
                # 添加搜索条件
                if keyword:
                    query += " AND (question ILIKE %s OR answer ILIKE %s OR category ILIKE %s)"
                    params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                count = result[0] if result else 0
                logger.info(f"Counted {count} FAQs")
                return count
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error counting FAQs: {e}")
            return 0
    
    def fetch_faq_by_id(self, faq_id: int) -> Optional[Dict]:
        try:
            conn = self.connect()
            cursor = conn.cursor(cursor_factory=DictCursor)
            try:
                cursor.execute(f"""
                    SELECT id, question, answer, category, sort, status, create_time
                    FROM {settings.DB_SCHEMA}.faq
                    WHERE id = %s
                """, (faq_id,))
                result = cursor.fetchone()
                if result:
                    # 将DictCursor对象转换为普通字典
                    faq = dict(result)
                    logger.info(f"Fetched FAQ with id: {faq_id}")
                    return faq
                logger.info(f"No FAQ found with id: {faq_id}")
                return None
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error fetching FAQ by id: {e}")
            return None
    
    def create_faq(self, faq: Dict) -> bool:
        """创建FAQ"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    INSERT INTO {settings.DB_SCHEMA}.faq 
                    (question, answer, category, sort, status, create_by, create_time, update_by, update_time) 
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s, CURRENT_TIMESTAMP)
                """, (
                    faq.get('question'),
                    faq.get('answer'),
                    faq.get('category'),
                    faq.get('sort', 0),
                    faq.get('status', 1),
                    faq.get('create_by'),
                    faq.get('update_by')
                ))
                conn.commit()
                logger.info(f"Created FAQ: {faq.get('question')}")
                return True
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error creating FAQ: {e}")
            return False
    
    def update_faq(self, faq_id: int, faq: Dict) -> bool:
        """更新FAQ"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    UPDATE {settings.DB_SCHEMA}.faq 
                    SET question = %s, 
                        answer = %s, 
                        category = %s, 
                        sort = %s, 
                        status = %s, 
                        update_by = %s, 
                        update_time = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (
                    faq.get('question'),
                    faq.get('answer'),
                    faq.get('category'),
                    faq.get('sort', 0),
                    faq.get('status', 1),
                    faq.get('update_by'),
                    faq_id
                ))
                conn.commit()
                logger.info(f"Updated FAQ with id: {faq_id}")
                return True
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error updating FAQ: {e}")
            return False
    
    def delete_faq(self, faq_id: int) -> bool:
        """删除FAQ"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    DELETE FROM {settings.DB_SCHEMA}.faq 
                    WHERE id = %s
                """, (faq_id,))
                conn.commit()
                logger.info(f"Deleted FAQ with id: {faq_id}")
                return True
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error deleting FAQ: {e}")
            return False
    
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
