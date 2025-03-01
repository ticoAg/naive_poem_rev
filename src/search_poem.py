from asyncio import Semaphore
import asyncio
import os

import json
import random
from typing import Any

from openai import AsyncOpenAI
from pymilvus import DataType, MilvusClient
from milvus_model.hybrid.bge_m3 import BGEM3EmbeddingFunction
from milvus_model.dense.openai import OpenAIEmbeddingFunction
import numpy as np
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from tqdm import tqdm

from db.duckdb_manager import DuckDBCache
from utils.tools import timer_decorator
from loguru import logger

load_dotenv("/Users/ticoag/Documents/myws/naive_poem_rev/.env")

api_sem = Semaphore(10)


@timer_decorator
def load_data() -> Dataset:
    data_list = load_dataset("ticoAg/cotinus-poem")
    return data_list["train"]


async def vectorize_text(texts: list[str]) -> list:
    def chunked_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    embed_model = os.getenv("DEFAULT_EMBEDDING_MODEL")
    vectors = []
    chunk_size = 2048
    for text_chunk in tqdm(chunked_list(texts, chunk_size), total=(len(texts) // chunk_size) + 1):
        response = await async_openai_client.embeddings.create(input=text_chunk, model=embed_model)
        new_vectors = [i.embedding for i in response.data]
        vectors.extend(new_vectors)
    # embed_cache_db = DuckDBCache(db_path=":embed_cache:", table_name="embed_cache")
    # # 遍历每个分块
    # for text_chunk in tqdm(chunked_list(texts, chunk_size), total=(len(texts) // chunk_size) + 1):
    #     chunk_vectors = []
    #     texts_to_embed = []

    #     # 检查缓存并记录索引
    #     cached_indices = []
    #     uncached_indices = []
    #     for idx, text in enumerate(text_chunk):
    #         cached_vector = embed_cache_db.get_cached_result(text)
    #         if cached_vector:
    #             chunk_vectors.append(cached_vector)
    #             cached_indices.append(idx)
    #         else:
    #             chunk_vectors.append(None)  # 占位符，确保顺序一致
    #             texts_to_embed.append(text)
    #             uncached_indices.append(idx)

    #     # 如果有未缓存的文本，调用 API
    #     if texts_to_embed:
    #         response = await async_openai_client.embeddings.create(input=texts_to_embed, model=embed_model)
    #         new_vectors = [i.embedding for i in response.data]

    #         # 将新向量存储到缓存
    #         for text, vector in zip(texts_to_embed, new_vectors):
    #             embed_cache_db.cache_result(text, vector)

    #         # 将新向量填充到占位符位置
    #         for idx, vector in zip(uncached_indices, new_vectors):
    #             chunk_vectors[idx] = vector

    #     # 确保 chunk_vectors 与 text_chunk 一一对应
    #     vectors.extend(chunk_vectors)

    return vectors


def flatten_paragraphs(batch):
    """
    Flatten paragraphs for a batch of data items using NumPy for parallel processing.

    Args:
        batch (dict): A batch of data where each key maps to a list of values.
                      For example, {"id": [1, 2], "content": [[...], [...]]}.

    Returns:
        dict: A dictionary with flattened paragraphs for the entire batch.
    """
    flattened = {k: [] for k in batch}
    for i in range(len(batch["title"])):
        data_item = {key: batch[key][i] for key in batch}
        _dict = {k: v for k, v in data_item.items() if k != "content"}
        paragraphs = [para for sub_chapter in data_item["content"] for para in sub_chapter["paragraphs"]]
        num_paragraphs = len(paragraphs)
        for key, value in _dict.items():
            flattened[key].extend([value] * num_paragraphs)
        flattened["content"].extend(paragraphs)
    return flattened


@timer_decorator
async def vec_texts() -> list[dict]:
    ori_data: Dataset = load_data()
    ori_data = ori_data.select(random.sample([i for i in range(ori_data.num_rows)], 200))
    single_para_data = ori_data.map(flatten_paragraphs, batched=True, num_proc=os.cpu_count()).filter(
        lambda x: len(x["content"]) >= 8
    )  # 过滤单句小于8的句子

    text = single_para_data["content"]
    vectors: list = await vectorize_text(text)
    datas = []
    for idx, (data, vector) in enumerate(zip(single_para_data, vectors)):
        data["vector"] = vector
        data["id"] = idx
        if data["types"] == None:
            data["types"] = []
        datas.append(data)
    return datas
    # with open(".cache/TangShi_vector.json", "w", encoding="utf-8") as file_obj:
    #     json.dump(single_para_data, file_obj, ensure_ascii=False, indent=4)  # type: ignore


def create_collection():
    if milvus_client.has_collection(collection_name):
        logger.info(f"Collection {collection_name} already exists")
        try:
            milvus_client.drop_collection(collection_name)
            logger.success(f"Deleted the collection {collection_name}")
        except Exception as e:
            logger.exception(f"Error occurred while dropping collection: {e}")

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True, description=collection_name)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="dynasty", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="theme", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="appreciation", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="rhythmic", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="types", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=32, max_length=32)

    try:
        milvus_client.create_collection(collection_name=collection_name, schema=schema, shards_num=2)
        logger.success(f"Created Collection: {collection_name}")
    except Exception as e:
        logger.exception(f"Error occurred while creating collection: {e}")

    collection_info = milvus_client.describe_collection(collection_name)
    print(f"Collection Info: \n{collection_info}")


def insert_vec_to_collection(data: list[dict]):
    print(f"正在将数据插入集合：{collection_name}")
    res = milvus_client.insert(collection_name=collection_name, data=data)
    print(f"插入的实体数量: {res['insert_count']}")


def create_index():
    # 创建IndexParams对象，用于存储索引的各种参数
    index_params = milvus_client.prepare_index_params()
    # 设置索引名称
    vector_index_name = "vector_index"
    # 设置索引的各种参数
    index_params.add_index(
        # 指定为"vector"字段创建索引
        field_name="vector",
        # 设置索引类型
        index_type="IVF_FLAT",
        # 设置度量方式
        metric_type="IP",
        # 设置索引聚类中心的数量
        params={"nlist": 128},
        # 指定索引名称
        index_name=vector_index_name,
    )

    print(f"开始创建索引：{vector_index_name}")

    # 创建索引
    milvus_client.create_index(
        # 指定为哪个集合创建索引
        collection_name=collection_name,
        # 使用前面创建的索引参数创建索引
        index_params=index_params,
    )

    # 验证索引
    indexes = milvus_client.list_indexes(collection_name=collection_name)
    print(f"列出创建的索引：{indexes}")

    print("*" * 50)

    # 查看索引详情
    index_details = milvus_client.describe_index(
        collection_name=collection_name,
        # 指定索引名称，这里假设使用第一个索引
        index_name="vector_index",
    )

    print(f"索引vector_index详情：{index_details}")


def load_collection():
    # 5 加载集合
    print(f"正在加载集合：{collection_name}")
    milvus_client.load_collection(collection_name=collection_name)

    # 验证加载状态
    print(milvus_client.get_load_state(collection_name=collection_name))


async def search_example():
    # 6 搜索

    # 打印向量搜索结果
    def print_vector_results(_res):
        """
        打印向量搜索结果。可变位置参数是输出的文本字段，不包括"distance"字段
        """
        # hit是搜索结果中的每一个匹配的实体
        _res: list[Any] = [hit["entity"] for hit in _res[0]]
        for item in _res:
            print(f"title: {item['title']}")
            print(f"author: {item['author']}")
            print(f"content: {item['content']}")
            print("-" * 50)
        print(f"数量：{len(_res)}")

    # 获取查询向量
    text = "今天的雨好大"
    # text = "我今天好开心"
    query_vectors = await vectorize_text([text])
    # 设置搜索参数
    search_params = {
        # 设置度量类型
        "metric_type": "IP",
        # 指定在搜索过程中要查询的聚类单元数量，增加nprobe值可以提高搜索精度，但会降低搜索速度
        "params": {"nprobe": 16},
    }

    # 指定如何输出搜索结果
    # 指定返回搜索结果的数量
    limit = 3
    # 指定返回的字段
    output_fields = ["author", "title", "content"]

    # 搜索案例1，向量搜索
    res1 = milvus_client.search(
        collection_name=collection_name,
        # 指定查询向量
        data=query_vectors,
        # 指定搜索的字段
        anns_field="vector",
        # 设置搜索参数
        search_params=search_params,
        limit=limit,
        output_fields=output_fields,
    )

    print(f"res1:")
    print(res1)
    print("=" * 100)
    print_vector_results(res1)
    print("=" * 100)

    # 搜索案例2，向量搜索，限制distance范围
    # 修改搜索参数，设置距离的范围
    search_params = {"metric_type": "IP", "params": {"nprobe": 16, "radius": 0.2, "range_filter": 1.0}}

    res2 = milvus_client.search(
        collection_name=collection_name,
        # 指定查询向量
        data=query_vectors,
        # 指定搜索的字段
        anns_field="vector",
        # 设置搜索参数
        search_params=search_params,
        # 删除limit参数
        output_fields=output_fields,
    )

    print(f"res2:")
    print(res2)
    print_vector_results(res2)
    print("=" * 100)

    # 搜索案例3，向量搜索+设置过滤条件
    _filter = f"author == '李白'"

    res3 = milvus_client.search(
        collection_name=collection_name,
        # 指定查询向量
        data=query_vectors,
        # 指定搜索的字段
        anns_field="vector",
        # 设置搜索参数
        search_params=search_params,
        limit=limit,
        filter=_filter,
        output_fields=output_fields,
    )

    print(f"res3:")
    print(res3)
    print_vector_results(res3)
    print("=" * 100)

    # 搜索案例4，标量搜索

    # 打印标量搜索结果
    def print_scalar_results(res):
        """
        打印标量搜索结果。可变位置参数是输出的文本字段，不包括"distance"字段
        """
        for hit in res:
            print(f"title: {hit['title']}")
            print(f"author: {hit['author']}")
            print(f"paragraphs: {hit['paragraphs']}")
            print("-" * 50)
        print(f"数量：{len(res)}")

    # 构建查询表达式1，包含指定文本
    _filter = f"paragraphs like '%雨%'"

    res4 = milvus_client.query(collection_name=collection_name, filter=_filter, output_fields=output_fields, limit=limit)

    print(f"res4:")
    print(res4)
    print_scalar_results(res4)
    print("=" * 100)

    # 搜索案例5，标量搜索
    # 构建查询表达式2，包含指定文本+设置过滤条件
    _filter = f"author == '李白' && paragraphs like '%雨%'"

    res5 = milvus_client.query(collection_name=collection_name, filter=_filter, output_fields=output_fields, limit=limit)

    print(f"res5:")
    print(res5)
    print_scalar_results(res5)
    print("=" * 100)


async def main():
    # data = await vec_texts()
    # create_collection()
    # insert_vec_to_collection(data)
    # create_index()
    load_collection()
    await search_example()


if __name__ == "__main__":
    # 创建client实例
    milvus_client = MilvusClient(uri="http://localhost:19530")
    async_openai_client = AsyncOpenAI()

    # 指定集合名称
    collection_name = "TangShi"
    asyncio.run(main())
