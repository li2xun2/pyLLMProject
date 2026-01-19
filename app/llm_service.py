from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import settings


class LLMService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"正在加载LLM模型: {settings.LLM_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.LLM_MODEL,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL,
                torch_dtype=torch.float16,
                device_map=settings.LLM_DEVICE,
                trust_remote_code=True
            )
            
            print(f"✓ LLM模型加载成功")
            print(f"  - 模型: {settings.LLM_MODEL}")
            print(f"  - 设备: {self.model.device}")
            print(f"  - 数据类型: {self.model.dtype}")
        except Exception as e:
            print(f"✗ LLM模型加载失败: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 128, temperature: float = 0.3) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("LLM模型未加载")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=False,
                top_p=0.9,
                top_k=30,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response = response[len(prompt):].strip()
        
        return response
    
    def generate_with_context(self, question: str, context: str, max_length: int = 512) -> str:
        prompt = f"参考信息：{context}\n\n问题：{question}\n\n根据参考信息回答："
        
        return self.generate(prompt, max_length=max_length, temperature=0.1)
    
    def chat(self, messages: List[dict], max_length: int = 512) -> str:
        prompt = self._format_chat_messages(messages)
        return self.generate(prompt, max_length=max_length, temperature=0.7)
    
    def _format_chat_messages(self, messages: List[dict]) -> str:
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


llm_service = LLMService()
