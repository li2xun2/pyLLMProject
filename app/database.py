import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
from app.config import settings


class Database:
    def __init__(self):
        self.connection = None
    
    def connect(self):
        if not self.connection or self.connection.closed:
            self.connection = psycopg2.connect(
                settings.DATABASE_URL,
                cursor_factory=RealDictCursor
            )
        return self.connection
    
    def disconnect(self):
        if self.connection and not self.connection.closed:
            self.connection.close()
    
    def get_all_tables(self) -> List[str]:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name NOT IN ('spatial_ref_sys', 'geography_columns', 'geometry_columns')
                ORDER BY table_name
            """)
            return [row['table_name'] for row in cursor.fetchall()]
        finally:
            cursor.close()
    
    def get_table_columns(self, table_name: str) -> List[Dict]:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = %s
                AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table_name,))
            return cursor.fetchall()
        finally:
            cursor.close()
    
    def fetch_table_data(self, table_name: str, text_columns: List[str], limit: int = None) -> List[Dict]:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            columns_str = ', '.join(text_columns)
            query = f"SELECT {columns_str} FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"
            
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
                    processed_results.append(processed_row)
            
            return processed_results
        finally:
            cursor.close()
    
    def fetch_all_faqs(self) -> List[Dict]:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id, question, answer FROM faq ORDER BY id")
            return cursor.fetchall()
        finally:
            cursor.close()
    
    def fetch_faq_by_id(self, faq_id: int) -> Optional[Dict]:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id, question, answer FROM faq WHERE id = %s", (faq_id,))
            return cursor.fetchone()
        finally:
            cursor.close()


db = Database()
