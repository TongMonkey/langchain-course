# 检索评分链。接受原始问题和检索到的文档，判断文档是否与问题相关，将会对检索到的每个文档执行此流程，相关的文档进行结构化输出，不相关的过滤掉

import os
from langchain_core.prompts import ChatPromptTemplate
# pydantic 是 python 里最常用的“数据结构+数据校验”的工具库
# BaseModel 是 pydantic 库中的一个类，用来定义数据结构
# Field 是 pydantic 库中的一个类，用来定义数据结构中的字段
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI

# temperature=0 表示使用最确定的输出
llm = AzureChatOpenAI(
    temperature=0,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

class GradeResponse(BaseModel):
    """Binary score to relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question， 'yes' or 'no'"
    )

# 使用 with_structured_output 方法，将 llm 包装成一个按照 GradeResponse 结构化输出的模型
# 当 LLM 被调用，会返回一个 GradeResponse 对象，里面包含 binary_score 字段，用来表示文档是否与问题相关
structured_llm_grader = llm.with_structured_output(GradeResponse)

# 定义一个系统提示词，用来告诉 LLM 如何判断文档是否与问题相关
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: {document}"),
        ("human", "User question: {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


