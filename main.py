from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from langchain_tavily import TavilySearch

# The point of this class here is to represent the source of the answer
class Source(BaseModel): 
    """Schema for a source used by the agent"""
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response of the agent"""
    answer: str = Field(description="The answer to the question")
    sources: List[Source] = Field(description="The list of sources to generate the answer")


api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv(
    "OPENAI_API_VERSION", "2024-12-01-preview"
)

llm = AzureChatOpenAI(
    temperature=0.6,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
    api_version=api_version,
    model=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-5.4-mini"),
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
