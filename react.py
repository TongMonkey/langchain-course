from dotenv import load_dotenv;
from langchain_core.tools import tool;
from langchain_openai import AzureChatOpenAI;
from langchain_tavily import TavilySearch;

load_dotenv();

@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3;

tools = [TavilySearch(max_results=1), triple]

llm = AzureChatOpenAI(
    model="gpt-5.4-mini",
    api_version="2024-12-01-preview",
    temperature=0,
).bind_tools(tools)