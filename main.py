from dotenv import load_dotenv
import os

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# TavilySearch is a tool that searches over internet built on top of the Tavily API witch is integrated with LangChain
from langchain_tavily import TavilySearch

# from tavily import TavilyClient 
# tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search results
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)


llm = ChatOpenAI(
    temperature=0.6,
    model="glm-4.6",
    openai_api_key=os.getenv("ZAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)
# tools = [search]
tools = [TavilySearch()] # 在 LangSmith 上看 trace 信息可以看到，被调用的 tool name is tavily_search, this is their naming convention
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Tokyo?")}
    )
    print(result)


if __name__ == "__main__":
    main()
