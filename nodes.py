from dotenv import load_dotenv;
from langchain_openai import AzureChatOpenAI;
from langgraph.graph import MessagesState;
from langgraph.prebuilt import ToolNode;

# 从 react.py 中导入 llm 和 tools
from react import llm, tools;
# Load environment variables from .env file
load_dotenv();



SYSTEM_MESSAGE = """
You are a helpful assistant that can use the following tools to answer questions:
{tools}
"""

# 定义一个节点，这个节点的作用是：根据当前的状态，调用 llm 模型，生成一条响应消息
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning loop.
    """
    response = llm.invoke(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                *state["messages"] # 把当前状态中(到这一刻为止的对话历史)的 messages 列表展开，作为 llm 的输入
            ]
        );
    return {"messages": [response]};

# 创建一个工具执行节点
tool_node = ToolNode(tools);