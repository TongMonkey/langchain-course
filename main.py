from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langchain_tavily import TavilySearch

# The point of this class here is to represent the source of the answer
class Source(BaseModel): 
    """Schema for a source used by the agent"""
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response of the agent"""
    answer: str = Field(description="The answer to the question")
    sources: List[Source] = Field(description="The list of sources to generate the answer")


llm = ChatOpenAI(
    temperature=0.6,
    model="glm-4.6",
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": HumanMessage(
                content="search for the top 3 job postings as an ai engineer in the last 30 days"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
