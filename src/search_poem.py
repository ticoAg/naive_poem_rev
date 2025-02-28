import os

import json
from typing import Any

from pymilvus import DataType, MilvusClient
from milvus_model.hybrid import BGEM3EmbeddingFunction
import numpy as np


def load_data() -> tuple[list[dict], list[str]]:
    with open("data/TangShi.json", "r", encoding="utf-8") as file:
        data_list = json.load(file)
        text = [data["paragraphs"][0] for data in data_list]
    return data_list, text


def vectorize_text(texts: list[str], model_name="BAAI/bge-large-zh-v1.5"):
    bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name, base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
    )
    embed_vectors: dict[str, np.ndarray[np.float32]] = bge_m3_ef.encode_documents(texts)
    return embed_vectors


def vec_texts():
    ori_data, text = load_data()
    vectors = vectorize_text(text)
    for data, vector in zip(ori_data, vectors["dense"]):
        data["vector"] = vector.tolist()
    with open(".cache/TangShi_vector.json", "w", encoding="utf-8") as file_obj:
        json.dump(ori_data, file_obj, ensure_ascii=False, indent=4)  # type: ignore


def create_collection():
    if client.has_collection(collection_name):
        print(f"Collection {collection_name} already exists")
        try:
            client.drop_collection(collection_name)
            print(f"Deleted the collection {collection_name}")
        except Exception as e:
            print(f"Error occurred while dropping collection: {e}")

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True, description="TangShi")
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="paragraphs", datatype=DataType.VARCHAR, max_length=10240)
    schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=128)

    try:
        client.create_collection(collection_name=collection_name, schema=schema, shards_num=2)
        print(f"Created collection {collection_name}")
    except Exception as e:
        print(f"Error occurred while creating collection: {e}")

    collection_info = client.describe_collection(collection_name)
    print(f"collection_info: {collection_info}")


def insert_vec_to_collection():
    with open(".cache/TangShi_vector.json", "r") as file:
        data = json.load(file)
        for item in data:
            item["paragraphs"] = item["paragraphs"][0]

    print(f"正在将数据插入集合：{collection_name}")
    res = client.insert(collection_name=collection_name, data=data)
    print(f"插入的实体数量: {res['insert_count']}")


def create_index():
    # 创建IndexParams对象，用于存储索引的各种参数
    index_params = client.prepare_index_params()
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
    client.create_index(
        # 指定为哪个集合创建索引
        collection_name=collection_name,
        # 使用前面创建的索引参数创建索引
        index_params=index_params,
    )

    # 验证索引
    indexes = client.list_indexes(collection_name=collection_name)
    print(f"列出创建的索引：{indexes}")

    print("*" * 50)

    # 查看索引详情
    index_details = client.describe_index(
        collection_name=collection_name,
        # 指定索引名称，这里假设使用第一个索引
        index_name="vector_index",
    )

    print(f"索引vector_index详情：{index_details}")


def load_collection():
    # 5 加载集合
    print(f"正在加载集合：{collection_name}")
    client.load_collection(collection_name=collection_name)

    # 验证加载状态
    print(client.get_load_state(collection_name=collection_name))


def search_example():
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
            print(f"paragraphs: {item['paragraphs']}")
            print("-" * 50)
        print(f"数量：{len(_res)}")

    # 获取查询向量
    text = "今天的雨好大"
    # text = "我今天好开心"
    query_vectors = [vectorize_text([text])["dense"][0].tolist()]

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
    output_fields = ["author", "title", "paragraphs"]

    # 搜索案例1，向量搜索
    res1 = client.search(
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

    res2 = client.search(
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

    res3 = client.search(
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

    res4 = client.query(collection_name=collection_name, filter=_filter, output_fields=output_fields, limit=limit)

    print(f"res4:")
    print(res4)
    print_scalar_results(res4)
    print("=" * 100)

    # 搜索案例5，标量搜索
    # 构建查询表达式2，包含指定文本+设置过滤条件
    _filter = f"author == '李白' && paragraphs like '%雨%'"

    res5 = client.query(collection_name=collection_name, filter=_filter, output_fields=output_fields, limit=limit)

    print(f"res5:")
    print(res5)
    print_scalar_results(res5)
    print("=" * 100)


if __name__ == "__main__":
    # 创建client实例
    client = MilvusClient(uri="http://localhost:19530")
    # 指定集合名称
    collection_name = "TangShi"

    # vec_texts()
    # create_collection()
    # insert_vec_to_collection()
    # create_index()
    load_collection()
    search_example()
