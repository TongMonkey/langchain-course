from typing import TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain


class MessageGraph(TypedDict):
    # 图的共享状态：messages 保存完整对话历史；add_messages 表示新消息会追加而不是覆盖。
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    # 生成节点：把历史消息交给生成链，产出下一版 tweet。
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    # 反思节点：根据目前历史生成 critique。
    res = reflect_chain.invoke({"messages": state["messages"]})
    # 包成 HumanMessage，让下一轮 generate 把 critique 当成“用户反馈”来改写。
    return {"messages": [HumanMessage(content=res.content)]}


# StateGraph 用 MessageGraph 作为状态结构，把函数节点连成可执行流程。
builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    # 用消息数量限制反思轮数，避免 generate <-> reflect 无限循环。
    if len(state["messages"]) > 6:
        return END
    return REFLECT


# generate 后动态判断：继续反思，或结束。
builder.add_conditional_edges(GENERATE, should_continue)
# reflect 后固定回到 generate，形成“批评 -> 改写”的循环。
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    # invoke 会从入口节点开始执行整张图，直到走到 END。
    response = graph.invoke(inputs)
    print(response)