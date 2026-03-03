from pydantic import BaseModel
from typing import List


class QARequest(BaseModel):
    documents: str
    questions: List[str]


