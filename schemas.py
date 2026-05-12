from typing import List

from pydantic import BaseModel, Field

# 这个反思类里，将会包含所有反思和批判的信息
class Reflection(BaseModel):
    # 缺失信息
    missing: str = Field(description="What is missing in the original answer?")
    # 冗余信息
    superfluous: str = Field(description="What is superfluous in the original answer?")

# 这个就是回答类，是我们希望从 LLM calls 获得的 output
class AnswerQuestion(BaseModel):
    answer: str = Field(description="The answer to the question.")
    reflection: Reflection = Field(description="The reflection on the answer.")
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")
