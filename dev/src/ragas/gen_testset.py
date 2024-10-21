import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from langchain_openai.llms.azure import AzureOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from ragas.testset.graph import Node
from ragas.testset.transforms.extractors import NERExtractor
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv("../../.env")

class TestsetGenerator:
    def __init__(self):
        self.llm = AzureOpenAI(
            azure_endpoint=