import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

# langchain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# azure_openai
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings

# local
from vectorstore import VectorStore

# load environment variables
load_dotenv("../../.env")


class RetrieverConfig(BaseModel):
    """
    リトリーバの設定

    Args:
        BaseModel (_type_): ベースモデル（Pydantic）
    """

    search_type: str = "similarity_score_threshold"  # 類似度を閾値で検索
    search_kwargs: Optional[dict] = {
        "score_threshold": 0.5,  # 類似度の閾値（以上）
        "k": 3,  # 検索結果の上位n件を返す
    }


class LangchainBot:
    """
    Langchainを利用したチャットボット
    """

    def __init__(
        self,
        retriever_config: Optional[RetrieverConfig] = RetrieverConfig(),
    ):
        """
        初期化

        Args:
            retriever_config (Optional[RetrieverConfig], optional): リトリーバの設定. Defaults to None.
        """
        # config
        self.retriever_config = retriever_config

        # model
        self.compose_llm = AzureChatOpenAI(
            deployment_name="gpt-4o-mini",
            azure_endpoint=os.getenv("AZURE_OPENA_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.stream_llm = AzureChatOpenAI(
            deployment_name="gpt-4o-mini",
            azure_endpoint=os.getenv("AZURE_OPENA_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small",
            azure_endpoint=os.getenv("AZURE_OPENA_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

        # vectorstore
        self.vectorstore_manager = VectorStore()
        self.vectorstore = self.vectorstore_manager.load("../../data/vectorstore")
        self.retriever = self.vectorstore.as_retriever(**vars(self.retriever_config))

        # prompt
        self.system_prompt = SystemMessagePromptTemplate.from_template(
            """
            以下の`context`の情報に基づいて、質問に回答してください。
            また`context`に関係のない質問は無視してください。

            context:
            {context}
            """
        )
        self.human_prompt = HumanMessagePromptTemplate.from_template("{question}")
        self.prompt = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.human_prompt]
        )

    def searching_context(self, query: str) -> str:
        """
        質問に対応するコンテキストを検索する

        Args:
            question (str): 質問

        Returns:
            str: コンテキスト
        """
        # search context
        context = ",".join(
            [content.page_content for content in self.retriever.invoke(query)]
        )
        print(context)
        return context

    def invoke(self, question: str = "こんにちは") -> str:
        """
        チャットボットを起動する

        Args:
            query (str, optional): ユーザーの入力. Defaults to "こんにちは".

        Returns:
            str: チャットボットの応答
        """

        # cpmpose query
        query = question  # 一旦そのまま

        # generate answer
        chain: Runnable = (
            {
                "context": self.searching_context,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.stream_llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        return answer


if __name__ == "__main__":
    bot = LangchainBot()
    print(bot.invoke("こんにちは"))
