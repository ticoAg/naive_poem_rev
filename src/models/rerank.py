# -*- encoding: utf-8 -*-
"""
@Time    :
@desc    :
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""
import os
from aiohttp import ClientSession

"""
payload = {
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "Apple",
    "documents": ["apple", "banana", "fruit", "vegetable"],
    "top_n": 4,
    "return_documents": False,
    "max_chunks_per_doc": 1024,
    "overlap_tokens": 80
}


response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)
"""


class ReRanker:
    def __init__(
        self,
        model_name: str = os.getenv("DEFAULT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
    ):
        self.model_name = model_name
        self.endpoint = os.getenv("OPENAI_BASE_URL") + "/rerank"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 4,
        return_documents: bool = False,
        model_name: str = None,
    ):
        assert query is not None, "query must be provided"
        assert documents is not None, "documents must be provided"
        assert top_n <= len(
            documents
        ), "top_n must be less than or equal to the number of documents"

        payload = {
            "model": self.model_name if not model_name else model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
            "max_chunks_per_doc": 1024,
            "overlap_tokens": 80,
        }
        async with ClientSession() as session:
            async with session.post(
                self.endpoint, json=payload, headers=self.headers, ssl=False
            ) as response:
                res = await response.json()
                return res["results"]
