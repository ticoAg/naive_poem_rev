from asyncio import Semaphore
import asyncio
import os

import random
from typing import Any, List

from openai import AsyncOpenAI
from pymilvus import DataType, MilvusClient

from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from tqdm import tqdm

from utils.tools import timer_decorator, chunked_list
from loguru import logger

load_dotenv()

api_sem = Semaphore(10)


@timer_decorator
def load_data() -> Dataset:
    data_list = load_dataset("ticoAg/cotinus-poem")
    return data_list["train"]


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
        paragraphs = [
            para
            for sub_chapter in data_item["content"]
            for para in sub_chapter["paragraphs"]
        ]
        num_paragraphs = len(paragraphs)
        for key, value in _dict.items():
            flattened[key].extend([value] * num_paragraphs)
        flattened["content"].extend(paragraphs)
    return flattened


async def process_chunk(
    idx: str, text_chunk: List[str], embed_model: str, semaphore: asyncio.Semaphore
) -> List:
    """
    处理单个文本分块，获取嵌入向量。
    使用信号量限制并发量。
    """
    async with semaphore:
        response = await async_openai_client.embeddings.create(
            input=text_chunk, model=embed_model
        )
        logger.debug(f"Processing chunk {idx} with {len(text_chunk)} texts")
        return [i.embedding for i in response.data]


async def vectorize_text(texts: List[str]) -> List:
    """
    异步并发处理文本列表，生成嵌入向量。
    """
    embed_model = os.getenv("DEFAULT_EMBEDDING_MODEL")
    vectors = []
    chunk_size = 1024
    max_concurrent_tasks = 32  # 控制最大并发任务数
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    text_chunks = chunked_list(texts, chunk_size)
    tasks = [
        process_chunk(
            "%s/%s" % (idx, len(texts) // chunk_size),
            text_chunk,
            embed_model,
            semaphore,
        )
        for idx, text_chunk in enumerate(text_chunks)
    ]
    results = await asyncio.gather(*tasks)  # 按任务提交顺序返回结果
    vectors = []
    for chunk_vectors in results:
        vectors.extend(chunk_vectors)  # 保证分块顺序
    return vectors


@timer_decorator
async def process_dataset() -> list[dict]:
    ori_data: Dataset = load_data()
    ori_data = ori_data.select(
        random.sample([i for i in range(ori_data.num_rows)], 100000)
    )
    single_para_data = ori_data.map(
        flatten_paragraphs, batched=True, num_proc=os.cpu_count()
    ).filter(
        lambda x: 8 <= len(x["content"]) <= 512
    )  # 过滤单句小于8的句子

    text = single_para_data["content"]
    vectors: list = await vectorize_text(text)
    datas = []
    for idx, (data, vector) in enumerate(zip(single_para_data, vectors)):
        data["vector"] = vector
        data["id"] = idx
        if data["types"] is None:
            data["types"] = []
        datas.append(data)
    return datas


def create_collection():
    if milvus_client.has_collection(collection_name):
        logger.info(f"Collection {collection_name} already exists")
        try:
            milvus_client.drop_collection(collection_name)
            logger.success(f"Deleted the collection {collection_name}")
        except Exception as e:
            logger.exception(f"Error occurred while dropping collection: {e}")

    schema = MilvusClient.create_schema(
        auto_id=False, enable_dynamic_field=True, description=collection_name
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="dynasty", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="theme", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(
        field_name="appreciation", datatype=DataType.VARCHAR, max_length=4096
    )
    schema.add_field(field_name="rhythmic", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=1536)
    schema.add_field(
        field_name="types",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=32,
        max_length=32,
    )

    try:
        milvus_client.create_collection(
            collection_name=collection_name, schema=schema, shards_num=2
        )
        logger.success(f"Created Collection: {collection_name}")
    except Exception as e:
        logger.exception(f"Error occurred while creating collection: {e}")

    collection_info = milvus_client.describe_collection(collection_name)
    print(f"Collection Info: \n{collection_info}")


def insert_vec_to_collection(data: list[dict], chunk_size: int = 256):
    """
    将数据分块插入到集合中。

    :param data: 需要插入的数据列表。
    :param chunk_size: 每个分块的大小。
    """
    ent_sum = 0
    for chunk in tqdm(
        chunked_list(data, chunk_size),
        desc="Insert data to collection: %s" % collection_name,
        total=len(data) // chunk_size,
    ):
        res = milvus_client.insert(collection_name=collection_name, data=chunk)
        ent_sum += res["insert_count"]
    logger.info(f"总插入的实体数量: {ent_sum}")


def create_index():
    index_params = milvus_client.prepare_index_params()
    vector_index_name = "vector_index"
    index_params.add_index(
        field_name="vector",  # 指定为"vector"字段创建索引
        index_type="IVF_FLAT",  # 设置索引类型
        metric_type="IP",  # 设置度量方式
        params={"nlist": 128},  # 设置索引聚类中心的数量
        index_name=vector_index_name,  # 指定索引名称
    )

    print(f"开始创建索引：{vector_index_name}")

    # 创建索引
    milvus_client.create_index(
        collection_name=collection_name,  # 指定为哪个集合创建索引
        index_params=index_params,  # 使用前面创建的索引参数创建索引
    )

    # 验证索引
    indexes = milvus_client.list_indexes(collection_name=collection_name)
    print(f"列出创建的索引：{indexes}")

    print("*" * 50)

    # 查看索引详情
    index_details = milvus_client.describe_index(
        collection_name=collection_name, index_name="vector_index"
    )

    print(f"索引vector_index详情：{index_details}")


def load_collection():
    # 5 加载集合
    print(f"正在加载集合：{collection_name}")
    milvus_client.load_collection(collection_name=collection_name)

    # 验证加载状态
    print(milvus_client.get_load_state(collection_name=collection_name))


@timer_decorator
async def search_example():
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

    @timer_decorator
    async def vec_search_example_1():
        # 搜索案例1，向量搜索
        res1 = milvus_client.search(
            collection_name=collection_name,
            data=query_vectors,  # 指定查询向量
            anns_field="vector",  # 指定搜索的字段
            search_params={
                "metric_type": "IP",  # 设置度量类型
                "params": {
                    "nprobe": 16
                },  # 指定在搜索过程中要查询的聚类单元数量，增加nprobe值可以提高搜索精度，但会降低搜索速度
            },  # 设置搜索参数
            limit=limit,
            output_fields=output_fields,
        )
        print_vector_results(res1)
        print("=" * 100)

    @timer_decorator
    async def vec_search_example_2(param):
        # 搜索案例2，向量搜索，限制distance范围
        res2 = milvus_client.search(
            collection_name=collection_name,  # 指定查询向量
            data=query_vectors,  # 指定搜索的字段
            anns_field="vector",  # 设置搜索参数
            search_params=param,  # 删除limit参数
            output_fields=output_fields,
        )
        print_vector_results(res2)
        print("=" * 100)

    @timer_decorator
    async def vec_search_example_3(param):
        # 搜索案例3，向量搜索+设置过滤条件
        _filter = f"author == '李白'"
        logger.debug("过滤条件: %s" % _filter)

        res3 = milvus_client.search(
            collection_name=collection_name,  # 指定查询向量
            data=query_vectors,  # 指定搜索的字段
            anns_field="vector",  # 设置搜索参数
            search_params=param,
            limit=limit,
            filter=_filter,
            output_fields=output_fields,
        )
        print_vector_results(res3)
        print("=" * 100)

    @timer_decorator
    async def scalar_search_example_1():
        # 搜索案例4，标量搜索
        _filter = f"paragraphs like '%雨%'"
        logger.info("过滤条件: %s" % _filter)

        res4 = milvus_client.query(
            collection_name=collection_name,
            filter=_filter,
            output_fields=output_fields,
            limit=limit,
        )
        print_scalar_results(res4)
        print("=" * 100)

    @timer_decorator
    async def scalar_search_example_2():
        # 搜索案例5，标量搜索
        # 构建查询表达式2，包含指定文本+设置过滤条件
        _filter = f"author == '李白' && paragraphs like '%雨%'"
        logger.debug("过滤条件: %s" % _filter)
        res5 = milvus_client.query(
            collection_name=collection_name,
            filter=_filter,
            output_fields=output_fields,
            limit=limit,
        )
        print_scalar_results(res5)
        print("=" * 100)

    text = "梅雨淅淅沥沥的下,有点冷"
    logger.info("Query: %s" % text)
    query_vectors = await vectorize_text([text])
    limit = 5
    output_fields = ["author", "title", "content"]

    await vec_search_example_1()

    # 修改搜索参数，设置距离的范围
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 16, "radius": 0.2, "range_filter": 1.0},
    }
    await vec_search_example_2(search_params)
    await vec_search_example_3(search_params)

    await scalar_search_example_1()
    await scalar_search_example_2()


async def main():
    # data = await process_dataset()
    # create_collection()
    # insert_vec_to_collection(data)
    # create_index()
    load_collection()
    await search_example()


if __name__ == "__main__":
    # 创建client实例
    milvus_client = MilvusClient(uri="http://localhost:19530")
    async_openai_client = AsyncOpenAI()

    collection_name = "poem"
    asyncio.run(main())
