"""
如何把「别的入参」传给 LangGraph 节点：三种常见写法（对照说明）。

运行：在项目根目录执行
  python customInputShowcase.py

本文件刻意不用真实 LLM，只演示状态与路由，避免依赖 .env。
"""

from __future__ import annotations

from functools import partial
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# 1) 写进 state：扩展图状态类型，invoke 时在初始 state 里带上字段
# ---------------------------------------------------------------------------
class StateWithExtras(TypedDict):
    """在 messages 之外增加只读/业务字段；每轮 invoke 由调用方写入。"""

    messages: Annotated[list, add_messages]
    tenant_id: str


def node_reads_state_extras(state: StateWithExtras) -> dict:
    tid = state["tenant_id"]
    last = state["messages"][-1]
    text = getattr(last, "content", str(last))
    reply = AIMessage(content=f"[tenant={tid}] echo: {text}")
    return {"messages": [reply]}


def build_graph_state_extra() -> StateGraph:
    g = StateGraph(StateWithExtras)
    g.add_node("work", node_reads_state_extras)
    g.set_entry_point("work")
    g.add_edge("work", END)
    return g.compile()


# ---------------------------------------------------------------------------
# 2) partial / 闭包：在 add_node 时把常量「绑」进可调用对象里
# ---------------------------------------------------------------------------
class MessagesOnlyState(TypedDict):
    messages: Annotated[list, add_messages]


def node_with_bound_multiplier(state: MessagesOnlyState, multiplier: int) -> dict:
    last = state["messages"][-1]
    text = getattr(last, "content", str(last))
    # 假装在做业务：把「配置」当成已绑好的 multiplier
    reply = AIMessage(content=f"scaled: {text!r} * {multiplier}")
    return {"messages": [reply]}


def build_graph_partial(multiplier: int):
    g = StateGraph(MessagesOnlyState)
    # LangGraph 仍只传 state；multiplier 由 partial 预先注入
    g.add_node("work", partial(node_with_bound_multiplier, multiplier=multiplier))
    g.set_entry_point("work")
    g.add_edge("work", END)
    return g.compile()


def build_graph_closure(multiplier: int):
    """闭包写法：与 partial 等价，只是把 multiplier 包在外层作用域里。"""

    def node_closed(state: MessagesOnlyState) -> dict:
        return node_with_bound_multiplier(state, multiplier)

    g = StateGraph(MessagesOnlyState)
    g.add_node("work", node_closed)
    g.set_entry_point("work")
    g.add_edge("work", END)
    return g.compile()


# ---------------------------------------------------------------------------
# 3) invoke(..., config=...)：每次调用可变的「运行期配置」
#    节点函数签名里增加 config: RunnableConfig（LangGraph 会注入）
# ---------------------------------------------------------------------------
class StateForConfig(TypedDict):
    messages: Annotated[list, add_messages]


def node_reads_runnable_config(
    state: StateForConfig, config: RunnableConfig
) -> dict:
    # 约定：可配置项放在 config["configurable"] 里
    configurable = config.get("configurable") or {}
    user_id = configurable.get("user_id", "anonymous")
    last = state["messages"][-1]
    text = getattr(last, "content", str(last))
    reply = AIMessage(content=f"[user_id={user_id}] got: {text}")
    return {"messages": [reply]}


def build_graph_config() -> StateGraph:
    g = StateGraph(StateForConfig)
    g.add_node("work", node_reads_runnable_config)
    g.set_entry_point("work")
    g.add_edge("work", END)
    return g.compile()


if __name__ == "__main__":
    print("--- 1) 额外字段在 state 里 ---")
    app1 = build_graph_state_extra()
    out1 = app1.invoke(
        {
            "tenant_id": "acme",
            "messages": [HumanMessage(content="hello")],
        }
    )
    print(out1["messages"][-1].content)

    print("--- 2a) functools.partial 绑定常量 ---")
    app2a = build_graph_partial(multiplier=10)
    out2a = app2a.invoke({"messages": [HumanMessage(content="7")]})
    print(out2a["messages"][-1].content)

    print("--- 2b) 闭包绑定常量（效果同 partial）---")
    app2b = build_graph_closure(multiplier=3)
    out2b = app2b.invoke({"messages": [HumanMessage(content="7")]})
    print(out2b["messages"][-1].content)

    print("--- 3) invoke 的 config[\"configurable\"] ---")
    app3 = build_graph_config()
    out3 = app3.invoke(
        {"messages": [HumanMessage(content="ping")]},
        config={"configurable": {"user_id": "u-42"}},
    )
    print(out3["messages"][-1].content)
