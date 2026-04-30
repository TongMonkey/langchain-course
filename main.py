from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph,END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

# 注册/定义节点名称
AGENT_REASON="agent_reason"
ACT= "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    # 在调用了 llm.invoke 生成消息后，如果最后一条消息带着 tools_calls 就返回 ACT ，没有 tool_calls，就返回 END
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

# 创建一个有状态的流程图，状态类型是 MessagesState，里面主要是 Messages 列表。
# 这个状态图的作用是：根据当前的状态，决定下一步要执行哪个节点
flow = StateGraph(MessagesState)

# 每当走到这个节点（AGENT_REASON），就执行 run_agent_reasoning 函数
flow.add_node(AGENT_REASON, run_agent_reasoning)

# 设置入口节点
flow.set_entry_point(AGENT_REASON)

# 到 ACT 节点，就执行 tool_node 函数
flow.add_node(ACT, tool_node)

# 设置条件边，先算一个函数 (should_continue) 根据返回值据欸的那个下一步是哪，如果是 END，就执行 END 节点，如果是 ACT，就执行 ACT 节点
# 这里的 END 是 LangGraph 的常量，表示流程结束。
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT
})

# 定义一个普通边，标识从 ACT 节点到 AGENT_REASON 节点的边, 没有分支
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)