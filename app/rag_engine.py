from typing import Dict, Optional, List
from app.database import db
from app.multi_vector_store import multi_vector_store
from app.llm_service import llm_service
from app.config import settings


class MultiTableRAGEngine:
    def __init__(self):
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return
        
        try:
            tables = settings.VECTOR_TABLES
            
            if not tables:
                tables = ['faq']
            
            for table_name in tables:
                if table_name in settings.VECTOR_TABLE_CONFIG:
                    text_columns = settings.VECTOR_TABLE_CONFIG[table_name]
                else:
                    columns_info = db.get_table_columns(table_name)
                    text_columns = [
                        col['column_name'] 
                        for col in columns_info 
                        if 'text' in col['data_type'].lower() or 
                           'char' in col['data_type'].lower() or
                           'varchar' in col['data_type'].lower()
                    ]
                
                if text_columns:
                    data = db.fetch_table_data(table_name, text_columns)
                    if data:
                        multi_vector_store.add_table_data(table_name, data, text_columns)
            
            self._initialized = True
            stats = multi_vector_store.get_all_stats()
            total_size = multi_vector_store.get_total_size()
            print(f"Multi-table RAG Engine initialized with {len(stats)} tables, {total_size} total records")
            for table_name, table_stats in stats.items():
                print(f"  - {table_name}: {table_stats['size']} records")
        except Exception as e:
            print(f"Error initializing Multi-table RAG Engine: {e}")
            raise
    
    def ask(self, question: str, tables: List[str] = None) -> Dict:
        if not self._initialized:
            self.initialize()
        
        if not question or not question.strip():
            return {
                "answer": "请提供有效的问题",
                "confidence": 0.0,
                "matched_question": None,
                "faq_id": None,
                "source": None,
                "url": None,
                "table": None
            }
        
        question_lower = question.lower()
        query_keywords = ['哪些', '所有', '列表', '全部', '有什么', '有哪些', '多少', '几个']
        is_query_type = any(keyword in question for keyword in query_keywords)
        
        customer_keywords = ['用户', '客户', '购买者', '消费者']
        order_keywords = ['订单', '购买', '交易', '配送', '发货']
        product_keywords = ['商品', '产品', '物品', '货物', '库存']
        
        target_tables = None
        if any(keyword in question for keyword in customer_keywords):
            target_tables = ['customers', 'faq']
        elif any(keyword in question for keyword in order_keywords):
            target_tables = ['orders', 'faq']
        elif any(keyword in question for keyword in product_keywords):
            target_tables = ['products', 'faq']
        
        if tables:
            target_tables = tables
        
        if settings.USE_LLM:
            return self._ask_with_llm(question, target_tables, is_query_type)
        else:
            return self._ask_without_llm(question, target_tables, is_query_type)
    
    def _ask_with_llm(self, question: str, target_tables: List[str], is_query_type: bool) -> Dict:
        results = multi_vector_store.search(question, tables=target_tables, top_k=3)
        
        if not results:
            return {
                "answer": "抱歉，未在数据库中找到相关信息。请尝试其他问题或联系客服。",
                "confidence": 0.0,
                "matched_question": None,
                "faq_id": None,
                "source": None,
                "url": None,
                "table": None
            }
        
        table_name = results[0][2]
        avg_confidence = sum(r[1] for r in results) / len(results)
        
        if is_query_type:
            formatted_results = []
            for result, confidence, _ in results:
                formatted_results.append(self._format_answer(result, table_name))
            
            if len(formatted_results) == 1:
                context = formatted_results[0]
            else:
                context = '\n\n'.join([f'{i+1}. {result}' for i, result in enumerate(formatted_results)])
        else:
            context = self._format_answer(results[0][0], table_name)
        
        if not context or not context.strip():
            return {
                "answer": "抱歉，未在数据库中找到相关信息。请尝试其他问题或联系客服。",
                "confidence": avg_confidence,
                "matched_question": None,
                "faq_id": None,
                "source": f"数据库表: {table_name}",
                "url": None,
                "table": table_name
            }
        
        try:
            answer = llm_service.generate_with_context(question, context)
            return {
                "answer": answer,
                "confidence": avg_confidence,
                "matched_question": results[0][0].get('question') or results[0][0].get('name'),
                "faq_id": results[0][0].get('id'),
                "source": f"数据库表: {table_name}",
                "url": None,
                "table": table_name
            }
        except Exception as e:
            print(f"LLM生成失败，使用原始答案: {e}")
            formatted_answer = self._format_answer(results[0][0], table_name)
            return {
                "answer": formatted_answer,
                "confidence": avg_confidence,
                "matched_question": results[0][0].get('question') or results[0][0].get('name'),
                "faq_id": results[0][0].get('id'),
                "source": f"数据库表: {table_name}",
                "url": None,
                "table": table_name
            }
    
    def _ask_without_llm(self, question: str, target_tables: List[str], is_query_type: bool) -> Dict:
        
        if is_query_type:
            results = multi_vector_store.search(question, tables=target_tables, top_k=10)
            if results:
                table_name = results[0][2]
                formatted_results = []
                for result, confidence, _ in results:
                    formatted_results.append(self._format_answer(result, table_name))
                
                if len(formatted_results) == 1:
                    answer = formatted_results[0]
                else:
                    answer = '\n\n'.join([f'{i+1}. {result}' for i, result in enumerate(formatted_results)])
                
                avg_confidence = sum(r[1] for r in results) / len(results)
                
                return {
                    "answer": answer,
                    "confidence": avg_confidence,
                    "matched_question": None,
                    "faq_id": None,
                    "source": f"数据库表: {table_name}",
                    "url": None,
                    "table": table_name
                }
            else:
                return {
                    "answer": "抱歉，未在数据库中找到相关信息。请尝试其他问题或联系客服。",
                    "confidence": 0.0,
                    "matched_question": None,
                    "faq_id": None,
                    "source": None,
                    "url": None,
                    "table": None
                }
        else:
            result, confidence, table_name = multi_vector_store.get_best_match(question, tables=target_tables)
            
            if result and confidence >= settings.CONFIDENCE_THRESHOLD:
                return {
                    "answer": self._format_answer(result, table_name),
                    "confidence": confidence,
                    "matched_question": result.get('question') or result.get('name'),
                    "faq_id": result.get('id'),
                    "source": f"数据库表: {table_name}",
                    "url": None,
                    "table": table_name
                }
            else:
                return {
                    "answer": "抱歉，未在数据库中找到相关信息。请尝试其他问题或联系客服。",
                    "confidence": confidence if result else 0.0,
                    "matched_question": None,
                    "faq_id": None,
                    "source": None,
                    "url": None,
                    "table": None
                }
    
    def _format_answer(self, result: Dict, table_name: str) -> str:
        if table_name == 'faq':
            return result.get('answer', '')
        elif table_name == 'orders':
            order_id = result.get('order_id', '')
            status = result.get('status', '')
            customer_name = result.get('customer_name', '')
            if customer_name:
                return f"{customer_name}的订单 {order_id} 状态为：{status}"
            else:
                return f"订单 {order_id} 状态为：{status}"
        elif table_name == 'customers':
            name = result.get('name', '')
            email = result.get('email', '')
            phone = result.get('phone', '')
            address = result.get('address', '')
            
            parts = []
            if email:
                parts.append(f"邮箱 {email}")
            if phone:
                parts.append(f"电话 {phone}")
            if address:
                parts.append(f"地址 {address}")
            
            if name and parts:
                return f"{name}的联系方式：{', '.join(parts)}"
            elif parts:
                return '、'.join(parts)
            else:
                return '未找到相关信息'
        elif table_name == 'products':
            name = result.get('name', '')
            description = result.get('description', '')
            category = result.get('category', '')
            
            parts = []
            if description:
                parts.append(description)
            if category:
                parts.append(f"属于{category}类别")
            
            if name and parts:
                return f"{name}：{', '.join(parts)}"
            elif name:
                return name
            else:
                return '未找到相关信息'
        else:
            parts = []
            for key, value in result.items():
                if not key.startswith('_') and value:
                    parts.append(f"{key}: {value}")
            return ' | '.join(parts)
    
    def refresh(self):
        self._initialized = False
        self.initialize()
    
    def get_stats(self) -> Dict:
        return {
            "initialized": self._initialized,
            "tables": multi_vector_store.get_all_stats(),
            "total_records": multi_vector_store.get_total_size(),
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "top_k": settings.TOP_K
        }


rag_engine = MultiTableRAGEngine()
