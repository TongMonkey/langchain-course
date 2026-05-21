# Will hold the graph of the nodes and edges, and connections between them

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from graph.consts import *
from graph.nodes import *
from graph.state import GraphState

load_dotenv()

def deside_to_generate(state: GraphState) -> bool:
    if state['web_search']:
        return WEBSEARCH
    else:
        return GENERATE

# 创建工作流图, 用来连接各个节点和边
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.add_edge(START, RETRIEVE) # 跟 workflow.set_entry_point(RETRIEVE) 一样
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(GRADE_DOCUMENTS, deside_to_generate, {WEBSEARCH: GENERATE, GENERATE: GENERATE})

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file="workflow.png")