from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info(f"Loading LLM model: {settings.LLM_MODEL}")
            
            # 自动检测CUDA是否可用
            actual_device = settings.LLM_DEVICE
            if actual_device == "cuda" and not torch.cuda.is_available():
                actual_device = "cpu"
                logger.warning("CUDA is not available, falling back to CPU")
            
            # 加载tokenizer
            logger.info(f"Loading tokenizer from: {settings.LLM_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.LLM_MODEL,
                trust_remote_code=True,
                local_files_only=True,
                encoding="utf-8"
            )
            logger.info("Tokenizer loaded successfully")
            
            # 加载模型
            logger.info(f"Loading model from: {settings.LLM_MODEL}")
            logger.info(f"Using device: {actual_device}")
            
            # 对于CPU，使用float32以避免精度问题
            if actual_device == "cpu":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL,
                torch_dtype=torch_dtype,
                device_map=actual_device if actual_device != "cuda" else "auto",
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 对于Qwen模型的特殊处理
            if "qwen" in settings.LLM_MODEL.lower():
                logger.info("Applying Qwen-specific optimizations")
                # 只有在GPU上才使用FP16
                if actual_device == "cuda" and hasattr(self.model, "half") and torch.cuda.is_available():
                    self.model = self.model.half()
            
            # 移动到正确的设备
            if actual_device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            elif actual_device == "cpu":
                self.model = self.model.cpu()
            
            logger.info(f"✓ LLM model loaded successfully")
            logger.info(f"  - Model: {settings.LLM_MODEL}")
            logger.info(f"  - Device: {self.model.device}")
            logger.info(f"  - Data type: {self.model.dtype}")
        except Exception as e:
            logger.error(f"✗ Failed to load LLM model: {e}")
            # 尝试使用CPU作为后备
            try:
                logger.info("Trying to load model on CPU as fallback")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    settings.LLM_MODEL,
                    trust_remote_code=True,
                    local_files_only=True,
                    encoding="utf-8"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.LLM_MODEL,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    local_files_only=True
                )
                logger.info("✓ Model loaded on CPU as fallback")
            except Exception as fallback_e:
                logger.error(f"✗ Fallback to CPU also failed: {fallback_e}")
                raise
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None and self.tokenizer is not None
    
    def generate(self, prompt: str, max_length: int = 128, temperature: float = 0.3) -> str:
        if not self.model or not self.tokenizer:
            logger.error("LLM model not loaded")
            raise RuntimeError("LLM模型未加载")
        
        try:
            logger.info(f"Starting to generate response for prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Starting to generate response for prompt: {prompt}")
            
            # 对于Qwen模型，使用其推荐的prompt格式
            if "qwen" in settings.LLM_MODEL.lower():
                messages = [
                    {"role": "system", "content": "你是一个专业的商城客服助手，需要根据用户的问题和提供的参考信息，给出准确、简洁的回答。回答要使用中文。"},
                    {"role": "user", "content": prompt}
                ]
                logger.info("Using Qwen chat template")
                
                # 先检查tokenizer是否有apply_chat_template方法
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    logger.info("Tokenizer has apply_chat_template method")
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )
                    
                    # 检查inputs的类型
                    if hasattr(inputs, 'input_ids'):
                        # BatchEncoding对象
                        logger.info(f"Chat template applied successfully, input_ids shape: {inputs.input_ids.shape}")
                        # 直接移动BatchEncoding对象到设备
                        if hasattr(inputs, 'to'):
                            inputs = inputs.to(self.model.device)
                            logger.info(f"BatchEncoding inputs moved to device: {self.model.device}")
                        else:
                            # 对于不支持to方法的BatchEncoding，手动移动每个张量
                            for key, value in inputs.items():
                                if hasattr(value, 'to'):
                                    inputs[key] = value.to(self.model.device)
                            logger.info(f"BatchEncoding inputs items moved to device: {self.model.device}")
                    else:
                        # 直接的张量
                        logger.info(f"Chat template applied successfully, input shape: {inputs.shape}")
                        inputs = inputs.to(self.model.device)
                        logger.info(f"Tensor inputs moved to device: {self.model.device}")
                else:
                    logger.error("Tokenizer does not have apply_chat_template method")
                    return "抱歉，系统暂时无法生成回答，请稍后再试。"
            else:
                logger.info("Using regular tokenization")
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                logger.info(f"Regular tokenization successful, input keys: {list(inputs.keys())}")
            
            logger.info(f"Generating response with max_length={max_length}, temperature={temperature}")
            
            with torch.no_grad():
                logger.info("Starting model.generate")
                
                # 准备正确的输入格式
                model_inputs = inputs
                if hasattr(inputs, 'input_ids'):
                    # 对于BatchEncoding对象，直接使用input_ids
                    logger.info("Using input_ids from BatchEncoding")
                    model_inputs = inputs.input_ids
                    logger.info(f"Input_ids shape: {model_inputs.shape}")
                
                outputs = self.model.generate(
                    model_inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                logger.info(f"Model.generate completed, output shape: {outputs.shape}")
            
            # 解码输出，确保使用正确的编码
            logger.info("Starting tokenizer.decode")
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            logger.info(f"Tokenizer.decode completed, response length: {len(response)}")
            logger.debug(f"Raw decoded response: {response[:100]}..." if len(response) > 100 else f"Raw decoded response: {response}")
            
            # 对于Qwen模型，提取助手的回答
            if "qwen" in settings.LLM_MODEL.lower():
                logger.info("Processing Qwen-specific response")
                # 使用更可靠的方式提取助手回答
                # 查找助手标记
                if "assistant" in response:
                    # 找到最后一个assistant标记
                    assistant_pos = response.rfind("assistant")
                    if assistant_pos != -1:
                        # 提取assistant后面的内容
                        response = response[assistant_pos + len("assistant"):].strip()
                        logger.info(f"Extracted assistant response after 'assistant' marker")
                # 如果还是没有找到，尝试查找其他常见标记
                elif "Assistant" in response:
                    assistant_pos = response.rfind("Assistant")
                    if assistant_pos != -1:
                        response = response[assistant_pos + len("Assistant"):].strip()
                        logger.info(f"Extracted assistant response after 'Assistant' marker")
                # 清理可能的乱码和特殊字符
                response = self._clean_response(response)
                logger.info(f"Cleaned Qwen response: {response[:100]}..." if len(response) > 100 else f"Cleaned Qwen response: {response}")
            else:
                response = response[len(prompt):].strip()
                logger.info(f"Extracted non-Qwen response: {response[:100]}..." if len(response) > 100 else f"Extracted non-Qwen response: {response}")
            
            # 确保返回的是有效的UTF-8字符串
            response = self._ensure_valid_utf8(response)
            logger.info(f"Final response: {response[:100]}..." if len(response) > 100 else f"Final response: {response}")
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # 返回一个默认的错误消息
            return "抱歉，系统暂时无法生成回答，请稍后再试。"
    
    async def generate_stream(self, prompt: str, max_length: int = 128, temperature: float = 0.3):
        """异步流式生成回答"""
        if not self.model or not self.tokenizer:
            logger.error("LLM model not loaded")
            yield {"error": "LLM模型未加载"}
            return
        
        try:
            logger.info(f"Starting stream generation for prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Starting stream generation for prompt: {prompt}")
            
            # 对于Qwen模型，使用其推荐的prompt格式
            if "qwen" in settings.LLM_MODEL.lower():
                messages = [
                    {"role": "system", "content": "你是一个专业的商城客服助手，需要根据用户的问题和提供的参考信息，给出准确、简洁的回答。回答要使用中文。"},
                    {"role": "user", "content": prompt}
                ]
                logger.info("Using Qwen chat template for streaming")
                
                # 先检查tokenizer是否有apply_chat_template方法
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    logger.info("Tokenizer has apply_chat_template method")
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )
                    
                    # 检查inputs的类型
                    if hasattr(inputs, 'input_ids'):
                        # BatchEncoding对象
                        logger.info(f"Chat template applied successfully, input_ids shape: {inputs.input_ids.shape}")
                        # 直接移动BatchEncoding对象到设备
                        if hasattr(inputs, 'to'):
                            inputs = inputs.to(self.model.device)
                            logger.info(f"BatchEncoding inputs moved to device: {self.model.device}")
                        else:
                            # 对于不支持to方法的BatchEncoding，手动移动每个张量
                            for key, value in inputs.items():
                                if hasattr(value, 'to'):
                                    inputs[key] = value.to(self.model.device)
                            logger.info(f"BatchEncoding inputs items moved to device: {self.model.device}")
                    else:
                        # 直接的张量
                        logger.info(f"Chat template applied successfully, input shape: {inputs.shape}")
                        inputs = inputs.to(self.model.device)
                        logger.info(f"Tensor inputs moved to device: {self.model.device}")
                else:
                    logger.error("Tokenizer does not have apply_chat_template method")
                    yield {"error": "抱歉，系统暂时无法生成回答，请稍后再试。"}
                    return
            else:
                logger.info("Using regular tokenization for streaming")
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                logger.info(f"Regular tokenization successful, input keys: {list(inputs.keys())}")
            
            logger.info(f"Stream generating with max_length={max_length}, temperature={temperature}")
            
            # 准备正确的输入格式
            model_inputs = inputs
            if hasattr(inputs, 'input_ids'):
                # 对于BatchEncoding对象，直接使用input_ids
                logger.info("Using input_ids from BatchEncoding for streaming")
                model_inputs = inputs.input_ids
                logger.info(f"Input_ids shape: {model_inputs.shape}")
            
            # 流式生成
            with torch.no_grad():
                logger.info("Starting model.generate_stream")
                
                # 直接生成完整回答，然后模拟流式输出
                # 这种方式虽然不是真正的流式生成，但可以避免token级别的错误
                try:
                    # 生成完整回答
                    outputs = self.model.generate(
                        model_inputs,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # 解码完整回答
                    full_output = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # 对于Qwen模型，需要提取助手的回答
                    if "qwen" in settings.LLM_MODEL.lower():
                        # 查找助手标记
                        if "assistant" in full_output:
                            assistant_pos = full_output.find("assistant")
                            full_output = full_output[assistant_pos + len("assistant"):].strip()
                        elif "Assistant" in full_output:
                            assistant_pos = full_output.find("Assistant")
                            full_output = full_output[assistant_pos + len("Assistant"):].strip()
                    else:
                        # 对于其他模型，跳过prompt部分
                        if len(full_output) > len(prompt):
                            full_output = full_output[len(prompt):].strip()
                    
                    # 清理可能的乱码和特殊字符
                    full_output = self._clean_response(full_output)
                    
                    # 确保返回的是有效的UTF-8字符串
                    full_output = self._ensure_valid_utf8(full_output)
                    
                    # 模拟流式输出，逐字返回
                    current_text = ""
                    for char in full_output:
                        current_text += char
                        yield {
                            "text": current_text,
                            "done": False
                        }
                        # 添加一个小延迟，模拟真实的流式效果
                        import asyncio
                        await asyncio.sleep(0.05)
                    
                    # 生成完成
                    logger.info("Stream generation completed")
                    yield {
                        "text": full_output,
                        "done": True
                    }
                except Exception as e:
                    # 捕获生成过程中的异常
                    logger.error(f"Error generating full response: {e}")
                    yield {"error": "抱歉，系统暂时无法生成回答，请稍后再试。"}
        except Exception as e:
            logger.error(f"Error in stream generation: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # 返回错误消息
            yield {"error": "抱歉，系统暂时无法生成回答，请稍后再试。"}
    
    def _clean_response(self, response: str) -> str:
        """清理响应文本，移除乱码和特殊字符"""
        # 移除控制字符
        import re
        response = re.sub(r'[\x00-\x1f\x7f]', '', response)
        # 移除多余的空白字符
        response = re.sub(r'\s+', ' ', response)
        return response.strip()
    
    def _ensure_valid_utf8(self, text: str) -> str:
        """确保文本是有效的UTF-8编码"""
        try:
            # 尝试编码和解码以确保有效性
            return text.encode('utf-8', 'replace').decode('utf-8')
        except Exception:
            return "抱歉，系统暂时无法生成回答，请稍后再试。"
    
    def generate_with_context(self, question: str, context: str, max_length: int = 512) -> str:
        """使用上下文生成回答，专为RAG场景优化"""
        try:
            # 确保输入文本是有效的UTF-8
            question = self._ensure_valid_utf8(question)
            context = self._ensure_valid_utf8(context)
            
            # 优化prompt格式，使其更适合Qwen模型
            if "qwen" in settings.LLM_MODEL.lower():
                prompt = f"参考信息：\n{context}\n\n用户问题：\n{question}\n\n请根据参考信息，直接回答用户问题，不要提及'参考信息'等引导性短语。回答要使用中文。"
            else:
                prompt = f"参考信息：{context}\n\n问题：{question}\n\n根据参考信息回答："
            
            # 生成回答
            answer = self.generate(prompt, max_length=max_length, temperature=0.1)
            
            # 确保回答是有效的UTF-8
            return self._ensure_valid_utf8(answer)
        except Exception as e:
            logger.error(f"Error in generate_with_context: {e}")
            return "抱歉，系统暂时无法生成回答，请稍后再试。"
    
    def chat(self, messages: List[dict], max_length: int = 512) -> str:
        """聊天模式，支持多轮对话"""
        try:
            # 对于Qwen模型，使用其专用的chat template
            if "qwen" in settings.LLM_MODEL.lower():
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_length,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 提取助手的回答
                if "assistant" in response:
                    assistant_start = response.rfind("assistant") + len("assistant")
                    response = response[assistant_start:].strip()
            else:
                # 对于其他模型，使用传统格式
                prompt = self._format_chat_messages(messages)
                response = self.generate(prompt, max_length=max_length, temperature=0.7)
            
            return response
        except Exception as e:
            logger.error(f"Error in chat mode: {e}")
            raise
    
    def _format_chat_messages(self, messages: List[dict]) -> str:
        """格式化聊天消息（非Qwen模型使用）"""
        formatted = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted += f"用户：{msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted += f"助手：{msg['content']}\n"
            elif msg["role"] == "system":
                formatted += f"系统：{msg['content']}\n"
        formatted += "助手："
        return formatted
    
    def get_status(self) -> dict:
        """获取模型状态信息"""
        if not self.model or not self.tokenizer:
            return {
                "loaded": False,
                "model": settings.LLM_MODEL,
                "error": "Model not loaded"
            }
        
        return {
            "loaded": True,
            "model": settings.LLM_MODEL,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "tokenizer_vocab_size": len(self.tokenizer)
        }


llm_service = LLMService()
