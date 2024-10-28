from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# local
from vectorstore import VectorStore


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


class SearchContext(BaseModel):
    """
    tool用のスキーマ

    Args:
        BaseModel (_type_): べースモデル（Pydantic）
    """

    query: str = Field(..., discription="質問")


@tool
def searching_context(query: str) -> str:
    """
    質問に対応するコンテキストを検索する
    Args:
        question (str): 質問
    Returns:
        str: コンテキスト
    """
    # search context
    retriever = (
        VectorStore().load("../../data/vectorstore").as_retriever(**RetrieverConfig())
    )
    context = ",".join([content.page_content for content in retriever.invoke(query)])
    print(context)  # debug
    return context
