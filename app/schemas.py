from pydantic import BaseModel, Field
from typing import Optional


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    tables: Optional[list] = Field(None, description="指定搜索的表名列表")


class AskResponse(BaseModel):
    answer: str
    confidence: float
    matched_question: Optional[str] = None
    faq_id: Optional[int] = None
    source: Optional[str] = None
    url: Optional[str] = None
    table: Optional[str] = None
