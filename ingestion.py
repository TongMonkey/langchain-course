import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("mediumblog.txt", encoding="utf-8")
    document = loader.load()

    print("splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embedding_deployment = (os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT") or "").strip()
    if not embedding_deployment:
        raise RuntimeError(
            "Set AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT in .env to the Azure deployment name of an "
            "embedding model (e.g. text-embedding-3-small). Chat deployments (gpt-*) are not valid "
            "for the embeddings API and cause DeploymentNotFound (404)."
        )

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=embedding_deployment,
    )

    index_name = os.environ.get("INDEX_NAME")
    if not index_name:
        raise RuntimeError("Set INDEX_NAME in .env.")

    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
    print("finish")
