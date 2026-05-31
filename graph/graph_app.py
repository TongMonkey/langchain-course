# Will hold the graph of the nodes and edges, and connections between them

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from graph.chains.question_router import question_router
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import *
from graph.nodes import *
from graph.state import GraphState

load_dotenv()

# 用来判断条件边的函数们：
def deside_to_generate(state: GraphState) -> bool:
    if state['web_search']:
        return WEBSEARCH_CONST
    else:
        return GENERATE_CONST


# 用 useful 和 not useful 来判断 generation 是否有用，从而决定要走向哪个 node: WEBSEARCH_CONST 或 END
# 用 "not supported" 来判断要不要重新生成答案
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:

    documents = state['documents']
    question = state['question']
    generation = state['generation']

    hallucination_score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    answer_score = answer_grader.invoke({"question": question, "generation": generation})

    hallucination_grade = hallucination_score.binary_score
    if hallucination_grade == 'yes':
        print("----DECISION: Generation is grounded in the documents----")
        print("----GRADE GENERATION vs QUESTION----")
        answer_grade = answer_score.binary_score
        if answer_grade == 'yes':
            print("----DECISION: Generation is grounded in the question----")
            return "useful"
        else:
            print("----DECISION: Generation is not grounded in the question----")
            return "not useful"
    else:
        print("Generation is not grounded in the documents")
        return "not supported"


def route_question(state: GraphState) -> str:
    print("----ROUTE QUESTION----")
    question = state['question']
    router_output = question_router.invoke({"question": question})
    if router_output.datasource == "vector_store":
        print("----DECISION: Route question to vector store----")
        return RETRIEVE_CONST
    else:
        print("----DECISION: Route question to web search----")
        return WEBSEARCH_CONST



# 创建工作流图, 用来连接各个节点和边
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE_CONST, retrieve)
workflow.add_node(GRADE_DOCUMENTS_CONST, grade_documents)
workflow.add_node(GENERATE_CONST, generate)
workflow.add_node(WEBSEARCH_CONST, web_search)

# 添加 conditional entry point, 用来根据 question 路由到不同的 node
workflow.set_conditional_entry_point(
    route_question,
    {
        RETRIEVE_CONST: RETRIEVE_CONST,
        WEBSEARCH_CONST: WEBSEARCH_CONST
    }
)
# 这行要删掉了，因为已经用 conditional entry point 代替了
# workflow.add_edge(START, RETRIEVE_CONST) # 跟 workflow.set_entry_point(RETRIEVE_CONST) 一样
workflow.add_edge(RETRIEVE_CONST, GRADE_DOCUMENTS_CONST)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS_CONST, 
    deside_to_generate, 
    {
        WEBSEARCH_CONST: GENERATE_CONST,
        GENERATE_CONST: GENERATE_CONST
    }
)

workflow.add_conditional_edges(
    GENERATE_CONST,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not useful": WEBSEARCH_CONST,
        "not supported": GENERATE_CONST
    }
)

workflow.add_edge(WEBSEARCH_CONST, GENERATE_CONST)
workflow.add_edge(GENERATE_CONST, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="workflow.png")