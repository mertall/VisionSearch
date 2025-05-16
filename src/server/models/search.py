from pydantic import BaseModel, Field
from typing import List

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Text query to search for")
    k: int = Field(default=5, ge=1, le=100, description="Top K results to return")

class SearchResult(BaseModel):
    image_path: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
