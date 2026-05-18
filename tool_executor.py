from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch

# 允许把 python function 转化为一个 Tool 以便 LLM 使用
from langchain_core.tools import StructuredTool, Tool

# ToolNode 是可以调用 tool 的节点，还会 look in the state for the messages key. 检查最后一条消息
# 如果找到，则会将 messages 作为参数传递给 tool。
from langgraph.prebuilt import ToolNode


from schemas import AnswerQuestion, ReviseAnswer


# 创建一个搜索对象,最多得到5个结果, 它将为我们提供一个具有搜索引擎功能的链工具
tavily_tool = TavilySearch(max_results=5)

# 运行查询，将接受一个 list of search queries 作为输入
# **kwargs 表示：除了已经写明的参数外，再允许任意「多余的关键字参数」，这里这么写是为了避免多传参数就报 TypeError 错误。
def run_queries(search_queries: list[str], **kwargs):
    """
    Run the generated queries.
    """
    # 将会并行执行所有的搜索查询 function
    return [tavily_tool.run(query) for query in search_queries]

# 这个工具生成的结果，就是 ToolMessage 类型，是工具执行回来的结果，会被 LangGraph 自动保存到 state["messages"] 中
execute_tools_node = ToolNode(
    [
        # LLM 发现消息里有叫 AnswerQuestion.__name__ 的 tool_calls, 就调用 run_queries 函数
        StructuredTool.from_function(
            func=run_queries,
            name=AnswerQuestion.__name__, # The name of the tool
            description="This tool is for the origin information",
        ),
        StructuredTool.from_function(
            func=run_queries,
            name=ReviseAnswer.__name__,
            description="This tool is for the revised information",
        ),
    ]
)