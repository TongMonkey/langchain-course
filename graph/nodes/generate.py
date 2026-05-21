# 用来生成最终答案的节点，负责执行 generation_chain

from typing import Any, Dict, List, Optional
from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # generation 是 LLM 给出的最终回答，是个 string
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}