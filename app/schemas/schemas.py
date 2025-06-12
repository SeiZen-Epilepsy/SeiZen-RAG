from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask the RAG system")
    
    @field_validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or only whitespace')
        return v.strip()

class SourceInfo(BaseModel):
    source_file: str = Field(..., description="Source file name")
    page: str = Field(..., description="Page number or identifier")
    content_snippet: str = Field(..., description="Relevant content snippet")
    relevance_score: Optional[float] = Field(None, description="Relevance score if available")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer from the RAG system")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source documents used")
    query_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional query metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict, description="Status of individual services")
    version: str = Field(..., description="Application version")
