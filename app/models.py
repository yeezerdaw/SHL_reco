from pydantic import BaseModel
from typing import List, Optional

class Assessment(BaseModel):
    name: str
    url: str
    remote_support: str
    adaptive_support: str
    duration: float
    test_type: str

class RecommendationRequest(BaseModel):
    query: str
    max_duration: Optional[int] = None

class RecommendationResponse(BaseModel):
    recommendations: List[Assessment]
