# Adaptive RAG

## Objective

在分支 project/xxt-self-rag 的基础上，添加一个问题导航功能，简单来说，就是用一个 question router to route question to different RAG flows.
所以改变之前的逻辑为:
判断用户的问题在 vector store 中是否已经存在，如果有，就直接 rag 查询,走向 retrieve node；否则就去 web search，走向 websearch node.

### Roles

question_router.py chain 
conditional entry: conditional edge with the first node of the entry point

