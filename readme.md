# Agentic RAG

## A demo of RAG with LangGraph

Reference to workflow-graph.png

### Steps

1. Fetch documents from given URLs
2. Ingest them into a vector store
3. Retrieve relevant documents with the user question
4. Grade the retrieved documents if they are relevant
5. Do web search if no document is relevant
6. Generate and output the final answer

### Roles

- node: 是图里的一个步骤，一个节点. 负责调用 chain, 把“做某件事”接到图的状态流里
- chain: 是“做事的逻辑单元”，可以理解成一个功能模块，通常表示一段可以被调用的流程：输入问题、调用 prompt、调用 LLM、解析输出，然后返回结果
- node 和 chain 的关系：node 通常会调用 chain