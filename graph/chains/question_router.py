from typing import Literal
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    # 用 datasource 这个字段来保存 vector store 里的值 或者 web search 里的值
    # Literal 是 Python 的类型提示，表示只能取给定的值中的一个, 所以输出的结果只能是 vector_store 或者 web_search
    datasource: Literal["vector_store", "web_search"] = Field(
        # python 语法里，Field 里加上省略号表示这个字段是必填的
        ..., 
        description="The datasource to use to answer the user's query"
    )



llm = AzureChatOpenAI(
    temperature=0,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router: RunnableSequence = route_prompt | structured_llm_router