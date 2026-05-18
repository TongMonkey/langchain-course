from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from chains import revisor, first_responder
from tool_executor import execute_tools_node

# 定义最大迭代次数
MAX_ITERATIONS = 2

# 定义第一个 node, 用来 draft first answer. 入参是 MessagesState, 第一条信息会是用户输入的propt 也就是 HumanMessage
# 这个 node 会调用 first_responder 链来生成第一个回答.
def draft_node(state: MessagesState) -> MessagesState:
    """Draft the initial response"""
    # 给一个叫 messages 的 key, 值是 state["messages"]
    response = first_responder.invoke({"messages": state["messages"]})
    # Append the message to the state
    return {"messages": [response]}


def revise_node(state: MessagesState) -> MessagesState:
    """Revise the response"""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

# 这个是事件循环节点，用来决定是否继续执行 execute_tools 节点。 如果工具调用次数大于最大迭代次数，则结束整个流程。
# 这里只为了学习，正常是应该由 LLM 来决定是否继续执行循环的
def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count."""
    # 统计工具调用次数，因为每次 Tool 调用都会往 state["messages"] 中添加一个 ToolMessage 类型，
    # 所以可以统计 ToolMessage 类型出现的次数, 从而计算 loop 循环的次数
    count_tool_visits = sum(
        # item 是 ToolMessage 类型
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    # 真是运行后，在 LangSmith 里可以看到，实际运行了 execute_tools 3次，所以 3 > 2 才停下的
    num_iterations = count_tool_visits
    # 如果工具调用次数大于最大迭代次数，则结束整个流程。
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools_node)
builder.add_node("revise", revise_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
# 用 event_loop 函数判断，是走向 execute_tools_node 还是 END node ，如果返回 "execute_tools"，则继续执行 execute_tools_node；如果返回 END，则结束整个流程。
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

graph = builder.compile()

# 会在控制台打印出一段文字，贴到 mermaid live 网站去，就能生成一个调用链条
print(graph.get_graph().draw_mermaid())

res = graph.invoke(
    {
        # 这个信息会被放到 MessagesPlaceholder(variable_name="messages") 中，比如 actor_prompt_template 里
        # 然后会被拼成 "SystemMessage:... 下面是从外面传进来的 messages"
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
            }
        ]
    }
)


last_message = res["messages"][-1]
# 如果最后一个信息是 AIMessage 类型且有 tool_calls 属性
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
