# Self RAG

## Objective

在分支 project/xxt-agentic-rag-graph 的基础上，在实现了一个 RAG 的基础上，再添加一个 reflection, 用于检查当前生成的答案 is hallucinated or not. 答案是否产生了幻觉。

### Roles

- hallucination_grader chain 
- answer_grader chain
- graph_app.py 调度者，在这里添加 conditional edge

#### hallucination_grader.py

A chain to determine whether the answer we get back from LLM, the generation is grounded in the documents.
创建一个 chain，用来判断 LLM 生成的答案，是否真的基于检索到的 documents，而不是模型自己编出来的。