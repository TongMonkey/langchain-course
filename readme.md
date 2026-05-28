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
  ```
  nodes/retrieve.py 调用 ingestion.retriever
  nodes/grade_documents.py 调用 chains/retrieval_grader.py
  nodes/generate.py 调用 chains/generation.py
  ```

### Structure

```
langchain-course/
  main.py                  # 程序入口，调用 LangGraph app

  ingestion.py             # 抓网页、切分文档、写入/读取 Chroma 向量库

  graph/
    graph_app.py           # LangGraph 工作流定义
    state.py               # GraphState，中间状态结构
    consts.py              # 节点名称常量

    nodes/
      retrieve.py          # 检索节点
      grade_documents.py   # 文档相关性评分节点
      generate.py          # 最终回答生成节点

    chains/
      retrieval_grader.py  # 判断文档是否相关的 LLM chain
      generation.py        # RAG answer generation chain
      web_search.py        # Tavily web search 工具节点
```

#### 入口

main.py 直接执行，在这个文件中定义了用户的问题

#### 数据准备层

ingestion.py 做了 RAG 的数据准备，在这个文件中定义了要抓取数据的三个 URLs

### 我的想法：

如果我想改个问题，或者想修改一下要抓取的地址，就要修改 main.py 和 ingestion.py 两个文件，这合理吗？是否要把这种用户输入的地方统一放在某个地方呢？

```
比如
// config.py
SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
user_question = "..."

然后在 main.py 和 ingestion.py 两个文件中都导入 config.py
```
