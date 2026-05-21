# 检索评分节点，用来检索文档，判断是否与用户问题相关，如果相关，则按照要求结构化输出，否则过滤掉

from typing import Any, Dict
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

