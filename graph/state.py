# state.py file will hold the state of the graph


from typing import TypedDict, List

# 定义一个类，用来存储 graph execution 的 state 数据
# 继承自 TypedDict，里面包含了所需的所有 state 数据, 用于 graph execution 时，传递 state 数据
# 参数 question：问题保存在 state 中，因为我们总是想 reference it 参照它
# 参数 generation：是 LLM 的生成结果，包括最终答案和一些中间推理过程
# 参数 web_search：boolean 类型值，表示是否需要进行 web search 去找一些额外的信息
# 参数 documents：肯定需要文档本身，检索到的文档 or search 结果中的文档，以便进行 RAG 检索
class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]