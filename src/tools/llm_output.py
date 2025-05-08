from pydantic import BaseModel
from typing import List


class ModelResponse(BaseModel):
    reasoning: str
    final_answer: str


class EntityList(BaseModel):
    entity_list: List[str]
