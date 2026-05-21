# 网络搜索链

from asyncio import QueueShutDown
from typing import List, Dict, Any
# 需要把 search results 转化成 LangChain 的 Document 对象
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState


web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("Running web search...")
    question = state["question"]
    documents = state["documents"]

    # 得到在一个数组里有三个结果的数组
    tavily_results = web_search_tool.invoke({"query": question})
    # 把数组里所有元素的内容提取出来，然后合并起来
    joined_tavily_results = "\n".join([tavily_result["content"] for tavily_result in tavily_results])
    # 把结果转化成 Document 对象
    web_results_document = Document(
        page_content=joined_tavily_results,
        metadata={
            "source": "tavily",
        }
    )
    if documents is not None:
        # 把结果添加到 documents 数组中
        documents.append(web_results_document)
    else:
        documents = [web_results_document]
    return {
        "documents": documents,
        "question": question,
    }




# 当这个文件被直接运行时，才会执行这个，常被用来放在文件底部进行一些测试代码、演示代码、debug 代码等
if __name__ == "__main__":
    web_search(
        state={
            "question": "agent memory",
            "documents": None,
        }
    )