import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS

# from langchain_openai.llms.azure import AzureOpenAI

from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings  # test

# from langchain_community.embeddings
from langchain.text_splitter import CharacterTextSplitter

# load environment variables
load_dotenv("../../.env")


class VectorStore:
    def __init__(self):
        print(os.getenv("AZURE_OPENA_ENDPOINT"))
        self.vectorstore = Optional[FAISS]
        self.embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small",
            azure_endpoint=os.getenv("AZURE_OPENA_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large"
        )

    class TextSplitConfig(BaseModel):
        separator: str = "。"
        chunk_size: int = 140
        chunk_overlap: int = 0

    def create(self, config: TextSplitConfig, text: str) -> FAISS:
        texts = self.text_split(text, config)  # split text into chunks
        docs = [Document(page_content=text) for text in texts]

        input(len(docs))

        self.vectorstore = FAISS.from_documents(
            documents=docs, embedding=self.embeddings
        )
        return self.vectorstore

    def text_split(self, text: str, config: TextSplitConfig) -> list[str]:
        splitter = CharacterTextSplitter(
            separator=config.separator,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        texts = splitter.split_text(text)
        return texts

    def save(self, save_path: str) -> None:
        self.vectorstore.save_local(folder_path=save_path)


if __name__ == "__main__":
    vs = VectorStore()
    config = vs.TextSplitConfig()

    with open("../../data/bocchi.txt", "r", encoding="utf-8") as f:
        text = f.read()

    vs.create(config=config, text=text)
    vs.save("../../data/vectorstore")

    print("Vector store created and saved.")
