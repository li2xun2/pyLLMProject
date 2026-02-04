from typing import Dict, Optional, List, Any
from app.database import db
from app.multi_vector_store import multi_vector_store
from app.llm_service import llm_service
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTableRAGEngine:
    def __init__(self):
        self._initialized = False
        # 存储对话历史，支持多轮对话
        self.conversation_history = {}
        # 对话历史最大长度
        self.max_history_length = 5
        # 系统提示词
        self.system_prompt = "你是一个专业的商城客服助手，需要根据用户的问题和提供的参考信息，给出准确、简洁、友好的回答。回答要直接针对用户的问题，不要提及'参考信息'等引导性短语。"
    
    def get_conversation_history(self, user_id: Optional[str] = None) -> List[dict]:
        """获取用户的对话历史"""
        if not user_id:
            return []
        return self.conversation_history.get(user_id, [])
    
    def add_to_history(self, user_id: Optional[str], role: str, content: str):
        """添加对话到历史记录"""
        if not user_id:
            return
        
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        # 添加新对话
        self.conversation_history[user_id].append({
            "role": role,
            "content": content
        })
        
        # 保持历史记录不超过最大长度
        if len(self.conversation_history[user_id]) > self.max_history_length:
            self.conversation_history[user_id] = self.conversation_history[user_id][-self.max_history_length:]
        
        logger.debug(f"Added {role} message to history for user {user_id}")
    
    def clear_history(self, user_id: Optional[str] = None):
        """清除对话历史"""
        if user_id and user_id in self.conversation_history:
            del self.conversation_history[user_id]
            logger.info(f"Cleared conversation history for user {user_id}")
        elif not user_id:
            self.conversation_history.clear()
            logger.info("Cleared all conversation history")
    
    def initialize(self, user_id: Optional[str] = None):
        if self._initialized and not user_id:
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
                    # 传递用户ID进行数据过滤
                    data = db.fetch_table_data(table_name, text_columns, user_id=user_id)
                    if data:
                        multi_vector_store.add_table_data(table_name, data, text_columns)
            
            self._initialized = True
            stats = multi_vector_store.get_all_stats()
            total_size = multi_vector_store.get_total_size()
            logger.info(f"Multi-table RAG Engine initialized with {len(stats)} tables, {total_size} total records")
            for table_name, table_stats in stats.items():
                logger.info(f"  - {table_name}: {table_stats['size']} records")
        except Exception as e:
            logger.error(f"Error initializing Multi-table RAG Engine: {e}")
            raise
    
    def ask(self, question: str, tables: List[str] = None, user_id: Optional[str] = None, conversation_id: Optional[str] = None) -> Dict:
        """处理用户问题并返回回答，确保正确编码"""
        if not self._initialized:
            self.initialize(user_id)
        
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
        
        # 确保问题是有效的UTF-8
        try:
            question = question.encode('utf-8', 'replace').decode('utf-8')
        except Exception:
            question = ""
        
        # 添加用户问题到对话历史
        self.add_to_history(user_id, "user", question)
        
        question_lower = question.lower()
        
        # 优化意图识别关键词
        query_keywords = ['哪些', '所有', '列表', '全部', '有什么', '有哪些', '多少', '几个', '查询', '查看']
        is_query_type = any(keyword in question for keyword in query_keywords)
        
        # 优化表选择逻辑
        customer_keywords = ['用户', '客户', '购买者', '消费者', '我', '我的信息', '个人信息', '账号', '个人资料', '会员信息', '注册信息', '登录信息', '我的账号', '我的会员']
        order_keywords = ['订单', '购买', '交易', '配送', '发货', '我的订单', '物流', '快递', '买过', '购买记录', '历史订单', '订单状态', '订单详情', '我的购买', '购买历史', '已购', '买了什么', '买了哪些', '购买过什么', '购买过哪些', '订单查询', '查订单', '查看订单', '订单列表', '所有订单', '全部订单']
        product_keywords = ['商品', '产品', '物品', '货物', '库存', '商品信息', '购买商品', '商品详情', '产品信息', '商品价格', '产品价格', '商品库存', '产品库存', '商品规格', '产品规格', '商品属性', '产品属性', '商品分类', '产品分类']
        address_keywords = ['地址', '收货地址', '我的地址', '地址管理', '收货信息', '地址列表', '添加地址', '修改地址', '删除地址', '默认地址', '地址详情', '我的收货地址']
        payment_keywords = ['支付', '付款', '账单', '退款', '退换货', '支付方式', '付款方式', '支付失败', '付款失败', '退款申请', '退款流程', '退款状态', '退换货流程', '退换货政策', '退款到账', '退款时间']
        
        target_tables = None
        if any(keyword in question for keyword in order_keywords):
            target_tables = ['oms_order']
        elif any(keyword in question for keyword in payment_keywords):
            target_tables = ['oms_order']
        elif any(keyword in question for keyword in customer_keywords):
            target_tables = ['ums_member']
        elif any(keyword in question for keyword in product_keywords):
            target_tables = ['pms_product']
        elif any(keyword in question for keyword in address_keywords):
            target_tables = ['ums_member_address']
        else:
            # 默认搜索用户和订单表
            target_tables = ['ums_member', 'oms_order']
        
        if tables:
            target_tables = tables
        
        logger.info(f"Processing question: '{question}' with tables: {target_tables}")
        
        if settings.USE_LLM:
            result = self._ask_with_llm(question, target_tables, is_query_type, user_id)
        else:
            result = self._ask_without_llm(question, target_tables, is_query_type, user_id)
        
        # 添加助手回答到对话历史
        self.add_to_history(user_id, "assistant", result.get("answer", ""))
        
        return result
    
    def _ask_with_llm(self, question: str, target_tables: List[str], is_query_type: bool, user_id: Optional[str] = None) -> Dict:
        """使用LLM处理问题，确保正确编码"""
        # 如果提供了user_id，并且查询涉及用户相关表，确保使用用户特定数据
        if user_id:
            # 对于用户相关的问题，获取用户的具体信息
            user_related_tables = ['orders', 'customers', 'member_address']
            if any(table in target_tables for table in user_related_tables):
                logger.info(f"Processing user-specific query for user {user_id}")
        
        # 搜索相关信息
        results = multi_vector_store.search(question, tables=target_tables, top_k=5)  # 增加搜索结果数量
        
        if not results:
            # 即使没有搜索结果，也尝试使用对话历史回答
            history = self.get_conversation_history(user_id)
            if len(history) > 1:
                # 使用对话历史进行回答
                try:
                    messages = [
                        {"role": "system", "content": self.system_prompt}
                    ] + history[:-1]  # 排除当前问题
                    messages.append({"role": "user", "content": question})
                    
                    answer = llm_service.chat(messages)
                    # 确保回答是有效的UTF-8
                    answer = answer.encode('utf-8', 'replace').decode('utf-8')
                    return {
                        "answer": answer,
                        "confidence": 0.5,
                        "matched_question": None,
                        "faq_id": None,
                        "source": "对话历史",
                        "url": None,
                        "table": None
                    }
                except Exception as e:
                    logger.error(f"Error using conversation history: {e}")
            
            return {
                "answer": "抱歉，未在数据库中找到相关信息。请尝试其他问题或联系客服。",
                "confidence": 0.0,
                "matched_question": None,
                "faq_id": None,
                "source": None,
                "url": None,
                "table": None
            }
        
        # 提取最相关的表和计算平均置信度
        table_name = results[0][2]
        avg_confidence = sum(r[1] for r in results) / len(results)
        
        # 构建上下文
        if is_query_type:
            # 对于查询类型问题，格式化所有相关结果
            formatted_results = []
            seen_results = set()  # 去重
            
            for result, confidence, _ in results:
                formatted = self._format_answer(result, table_name)
                if formatted and formatted not in seen_results:
                    seen_results.add(formatted)
                    formatted_results.append(formatted)
            
            if len(formatted_results) == 1:
                context = formatted_results[0]
            else:
                context = '\n\n'.join([f'{i+1}. {result}' for i, result in enumerate(formatted_results[:3])])  # 最多使用3个结果
        else:
            # 对于非查询类型问题，使用最相关的结果
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
            # 构建完整的上下文，包括用户信息和对话历史
            full_context = ""
            
            # 添加用户信息
            if user_id:
                user_profile = db.fetch_user_profile(user_id)
                if user_profile:
                    user_info = f"用户信息: 昵称={user_profile.get('nickname', '未知')}, 电话={user_profile.get('phone', '未知')}\n"
                    full_context += user_info
            
            # 添加对话历史（最近2轮）
            history = self.get_conversation_history(user_id)
            if len(history) > 2:
                recent_history = history[-3:-1]  # 最近2轮对话
                history_context = "最近对话:\n"
                for msg in recent_history:
                    if msg['role'] == 'user':
                        history_context += f"用户: {msg['content']}\n"
                    else:
                        history_context += f"助手: {msg['content']}\n"
                full_context += history_context + "\n"
            
            # 添加搜索结果
            full_context += f"参考信息:\n{context}"
            
            # 生成回答
            answer = llm_service.generate_with_context(question, full_context, max_length=1024)
            
            # 优化回答质量
            answer = self._optimize_answer(answer, question, user_id)
            
            # 确保所有字段都是有效的UTF-8
            answer = answer.encode('utf-8', 'replace').decode('utf-8')
            source = f"数据库表: {table_name}".encode('utf-8', 'replace').decode('utf-8')
            matched_question = results[0][0].get('question') or results[0][0].get('name')
            if matched_question:
                matched_question = matched_question.encode('utf-8', 'replace').decode('utf-8')
            
            return {
                "answer": answer,
                "confidence": avg_confidence,
                "matched_question": matched_question,
                "faq_id": results[0][0].get('id'),
                "source": source,
                "url": None,
                "table": table_name
            }
        except Exception as e:
            logger.error(f"LLM生成失败，使用原始答案: {e}")
            # 使用最相关的结果作为回答
            formatted_answer = self._format_answer(results[0][0], table_name)
            formatted_answer = formatted_answer.encode('utf-8', 'replace').decode('utf-8')
            source = f"数据库表: {table_name}".encode('utf-8', 'replace').decode('utf-8')
            matched_question = results[0][0].get('question') or results[0][0].get('name')
            if matched_question:
                matched_question = matched_question.encode('utf-8', 'replace').decode('utf-8')
            
            return {
                "answer": formatted_answer,
                "confidence": avg_confidence,
                "matched_question": matched_question,
                "faq_id": results[0][0].get('id'),
                "source": source,
                "url": None,
                "table": table_name
            }
    
    def _optimize_answer(self, answer: str, question: str, user_id: Optional[str] = None) -> str:
        """优化回答质量"""
        # 移除不必要的引导性短语
        unnecessary_phrases = [
            "根据参考信息，",
            "根据提供的信息，",
            "参考信息显示，",
            "参考信息表明，",
            "根据以上信息，",
            "综上所述，"
        ]
        
        for phrase in unnecessary_phrases:
            answer = answer.replace(phrase, "")
        
        # 确保回答直接针对问题
        if not answer.strip():
            answer = "抱歉，我无法回答这个问题。"
        
        # 添加礼貌用语
        polite_endings = [
            "如果您有其他问题，随时告诉我。",
            "还有什么可以帮您的吗？",
            "希望以上信息对您有所帮助。"
        ]
        
        # 随机选择一个礼貌结尾（20%概率）
        if len(answer) < 200:
            import random
            if random.random() < 0.2:
                answer += " " + random.choice(polite_endings)
        
        return answer.strip()
    
    def _ask_without_llm(self, question: str, target_tables: List[str], is_query_type: bool, user_id: Optional[str] = None) -> Dict:
        
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
        """格式化回答，确保正确编码"""
        try:
            if table_name == 'ums_member':
                nickname = result.get('nickname', '')
                phone = result.get('phone_hidden', '')
                city = result.get('city', '')
                province = result.get('province', '')
                
                parts = []
                if phone:
                    parts.append(f"电话 {phone}")
                if city:
                    parts.append(f"城市 {city}")
                if province:
                    parts.append(f"省份 {province}")
                
                if nickname and parts:
                    return f"{nickname}的信息：{', '.join(parts)}"
                elif parts:
                    return '、'.join(parts)
                else:
                    return '未找到相关信息'
            elif table_name == 'oms_order':
                order_sn = result.get('order_sn', '')
                status = result.get('status', '')
                receiver_name = result.get('receiver_name', '')
                total_amount = result.get('total_amount', '')
                
                status_map = {
                    '0': '待付款',
                    '1': '待发货',
                    '2': '待收货',
                    '3': '已完成',
                    '4': '已取消',
                    '5': '退款中',
                    '6': '已退款'
                }
                
                status_text = status_map.get(status, status)
                
                parts = []
                if receiver_name:
                    parts.append(f"收货人 {receiver_name}")
                if total_amount:
                    parts.append(f"金额 {total_amount}元")
                
                if order_sn and parts:
                    return f"订单 {order_sn} 状态为：{status_text}，{', '.join(parts)}"
                elif order_sn:
                    return f"订单 {order_sn} 状态为：{status_text}"
                else:
                    return '未找到相关信息'
            elif table_name == 'ums_member_address':
                name = result.get('name', '')
                phone = result.get('phone_hidden', '')
                province = result.get('province', '')
                city = result.get('city', '')
                district = result.get('district', '')
                address_detail = result.get('detail_address', '')
                
                address_parts = []
                if province:
                    address_parts.append(province)
                if city:
                    address_parts.append(city)
                if district:
                    address_parts.append(district)
                if address_detail:
                    address_parts.append(address_detail)
                
                full_address = ''.join(address_parts)
                
                parts = []
                if phone:
                    parts.append(f"电话 {phone}")
                if full_address:
                    parts.append(f"地址 {full_address}")
                
                if name and parts:
                    return f"{name}的收货信息：{', '.join(parts)}"
                elif parts:
                    return '、'.join(parts)
                else:
                    return '未找到相关信息'
            elif table_name == 'pms_product':
                name = result.get('name', '')
                brand_name = result.get('brand_name', '')
                product_category_name = result.get('product_category_name', '')
                detail_html = result.get('detail_html', '')
                
                parts = []
                if brand_name:
                    parts.append(f"品牌 {brand_name}")
                if product_category_name:
                    parts.append(f"类别 {product_category_name}")
                if detail_html:
                    # 提取纯文本并截断
                    import re
                    text = re.sub('<[^<]+?>', '', detail_html)
                    text = text.strip()[:100] + '...' if len(text) > 100 else text
                    parts.append(text)
                
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
        except Exception as e:
            logger.error(f"Error formatting answer: {e}")
            return '未找到相关信息'
    
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
