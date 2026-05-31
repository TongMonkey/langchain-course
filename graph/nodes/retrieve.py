# 实现一个检索节点，将提取用户的问题，利用向量库的搜索功能，检索出相关的文档，存入到 state

from typing import Dict, Any
from graph.state import GraphState
# from ingestion import retriever, vectorstore
from ingestion import retriever

# 返回一个字典，用于更新 state
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("REtrieve....")
    # Extract the question from the state
    question = state["question"]
    # Retrieve the documents from the vector store
    documents = retriever.invoke(question)
    # 这里的 question 其实不是必须返回的，只不过在 state 里自定义了个 question 属性，我们自己想存而已
    return {"documents": documents, "question": question}