from pydantic import BaseModel

class SourceInfo(BaseModel):
    file_name: str
    score: float
    snippet: str
    access_level: str
    
    
    
class QueryResult(BaseModel):
    answer: str
    sources: list[SourceInfo]
    escalate: bool