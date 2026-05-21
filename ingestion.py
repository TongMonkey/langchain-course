import os
from dotenv import load_dotenv
# 用递归字符分割器来分割文档
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 用基于 web 的加载器从互联网加载文档
from langchain_community.document_loaders import WebBaseLoader
# 用 chroma 这个开源的库做 vector store 矢量存储
from langchain_chroma import Chroma
# 用 langchain 的 openai 的 embeddings 模型来嵌入文档
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

# 一个要抓取数据的 URL 列表
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 用基于 web 的加载器从互联网加载文档,得到 文档列表 list of documents, 现在是一个双层嵌套的文档列表
docs = [WebBaseLoader(url).load() for url in urls]
# 把双层列表 扁平化
doc_list = [item for sublist in docs for item in sublist]


# 把文档分成小块
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
# 把文档分成小块, 这个是最终的各个小块的数组结果
doc_splits = text_splitter.split_documents(doc_list)

# 用 embedding 模型来把文本块转化成向量
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# 建库/写入数据库: 把小块存储到向量数据库中(磁盘上), 所以这里的 vectorstore 是 Chroma 对象
# 第一次执行后，之后再执行，会往这个相同的 persist_directory 中写入数据，覆盖掉原来的数据
# 所以后来注释掉了(项目文件夹下有了.chroma文件夹)，retriver 会从本地读取数据，如果删了.chroma文件夹，就需要重新建库
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    # 给数据库里这组数据起个名字
    collection_name="rag-chroma",
    # 用什么 embedding 模型来把每个文本块转化成向量，再存起来
    embedding=embeddings,
    # 添加参数：存储在哪，相对于 root 目录的相对路径
    persist_directory="./.chroma",
)

# 创建 retriever 检索器对象. 
# Chroma 对象本身是向量数据库对象，用来打开/读取已有向量库
# as_retriever() 方法把 Chroma 对象转化成 retriever 检索器对象
# 这个运行后，会在当前项目文件夹下多个 .chroma 文件夹，里面存储了向量数据库
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    # 用什么模型来把问题转换成向量，从而在向量数据库中进行匹配
    embedding_function=embeddings,
).as_retriever()

