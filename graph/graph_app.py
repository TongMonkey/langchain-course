# Will hold the graph of the nodes and edges, and connections between them

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import *
from graph.nodes import *
from graph.state import GraphState

load_dotenv()

# 用来判断条件边的函数们：
def deside_to_generate(state: GraphState) -> bool:
    if state['web_search']:
        return WEBSEARCH
    else:
        return GENERATE


# 用 useful 和 not useful 来判断 generation 是否有用，从而决定要走向哪个 node: WEBSEARCH 或 END
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



# 创建工作流图, 用来连接各个节点和边
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.add_edge(START, RETRIEVE) # 跟 workflow.set_entry_point(RETRIEVE) 一样
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS, 
    deside_to_generate, 
    {
        WEBSEARCH: GENERATE,
        GENERATE: GENERATE
    }
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not useful": WEBSEARCH,
        "not supported": GENERATE
    }
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="workflow.png")