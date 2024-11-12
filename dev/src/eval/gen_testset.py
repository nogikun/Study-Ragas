import os, sys
import pickle
import asyncio  # 非同期処理を行うため

from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from ragas.testset.graph import Node
from ragas.testset.transforms.extractors import (
    NERExtractor,
    SummaryExtractor,
    EmbeddingExtractor,
)
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms.relationship_builders.cosine import (
    SummaryCosineSimilarityBuilder,
    CosineSimilarityBuilder,
)

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import pandas as pd

# load environment variables

# 親ディレクトリのパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from vectorstore import VectorStore

load_dotenv("../../.env")


class KGTestsetGenerator:
    """
    _summary_
    """

    def __init__(
        self,
        vectorstore: Optional[FAISS] = None,
    ):
        """
        _summary_

        Args:
            vectorstore (Optional[FAISS], optional): _description_. Defaults to None.
        """
        self.llm = LangchainLLMWrapper(
            AzureChatOpenAI(
                deployment_name="gpt-4o-mini",
                azure_endpoint=os.getenv("AZURE_OPENA_ENDPOINT"),
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            )
        )
        self.embedding = LangchainEmbeddingsWrapper(
            AzureOpenAIEmbeddings(
                deployment="text-embedding-3-small",
                azure_endpoint=os.getenv("AZURE_OPENA_ENDPOINT"),
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            )
        )

        self.vectorstore = vectorstore

        # future
        self.testset: Optional[pd.DataFrame] = None

    async def gen_knowledge_graph(
        self,
        on_save: bool = False,
    ) -> pd.DataFrame:
        """
        _summary_

        Args:
            on_save (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        # initialize
        extractor = NERExtractor(llm=self.llm)
        summary_extractor = SummaryExtractor(llm=self.llm)
        embedding_extractor = EmbeddingExtractor(embedding_model=self.embedding)

        # ---

        # vectorstoreからコンテンツを抽出
        docs = list(self.vectorstore.docstore._dict.values())
        docs = docs[:10]  # テスト用に10件に制限

        input(type(docs))

        # ノードを生成
        nodes = []
        for doc in docs:
            node = Node(properties={"page_content": dict(doc).get("page_content")})
            nodes.append(node)

        # ner_extractor
        output = await asyncio.gather(*(extractor.extract(node) for node in nodes))
        _ = [
            node.properties.update({key: val})
            for (key, val), node in zip(output, nodes)
        ]  # プロパティを更新

        # # summary_extractor
        # output = await asyncio.gather(
        #     *(summary_extractor.extract(node) for node in nodes)
        # )
        # _ = [
        #     node.properties.update({key: val})
        #     for (key, val), node in zip(output, nodes)
        # ]  # プロパティを更新

        # embedding_extractor
        output = await asyncio.gather(
            *(embedding_extractor.extract(node) for node in nodes)
        )
        _ = [
            node.properties.update({key: val})
            for (key, val), node in zip(output, nodes)
        ]

        print(nodes[0].properties)

        if on_save:
            # save
            with open("../../data/knowledge_graph.pkl", "wb") as f:
                pickle.dump(nodes, f)

        kg = KnowledgeGraph(nodes=nodes)

        # rel_builder = JaccardSimilarityBuilder(
        #     property_name="entities",
        #     key_name="PER",
        #     new_property_name="entity_jaccard_similarity",
        # )

        rel_builder = CosineSimilarityBuilder(
            property_name="embedding",
            new_property_name="cosine_similarity",
            threshold=0.9,
        )
        self.relationships = await asyncio.gather(rel_builder.transform(kg))  # unused

        executor = TestsetGenerator(
            llm=self.llm, embedding_model=self.embedding, knowledge_graph=kg
        )
        self.testset = executor.generate_with_langchain_docs(
            documents=docs,
            testset_size=10,
            with_debugging_logs=True,
            raise_exceptions=False,
        ).to_pandas()  # pandas.DataFrame

        return self.testset


if __name__ == "__main__":
    vectorstore_manager = VectorStore()
    vectorstore = vectorstore_manager.load("../../data/vectorstore")
    generator = KGTestsetGenerator(vectorstore=vectorstore)
    testset = asyncio.run(generator.gen_knowledge_graph(on_save=True))
    print(testset)
    testset.to_csv("../../data/testset.csv", index=False)
