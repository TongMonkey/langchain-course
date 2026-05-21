# 在检索了数据后，可能也做了 web search, 然后就可以生成最终的回答了

import os
from langsmith import Client
# get the content and turn it to string
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI




llm = AzureChatOpenAI(
    temperature=0,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# 从 langChain 的 hub 里找一个现成的 promp template, 就不用自己写 ChatPromptTemplate.from_template(...)了
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
