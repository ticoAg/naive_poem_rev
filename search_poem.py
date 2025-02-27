# %pip install pymilvus "pymilvus[model]" torch

import torch
import json
from pymilvus import DataType, MilvusClient
from pymilvus.model.dense import OpenAIEmbeddingFunction


# 1 向量化文本数据
def vectorize_text(text, model_name="maidalun1020/bce-embedding-base_v1"):
	# 检查是否有可用的CUDA设备
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	# 根据设备选择是否使用fp16
	use_fp16 = device.startswith("cuda")
	# 创建嵌入模型实例
	bge_m3_ef = OpenAIEmbeddingFunction(
		model_name=model_name,
		base_url="http://localhost:7997/v1"
	)
	# 把输入的文本向量化
	vectors = bge_m3_ef.encode_documents(text)
	return vectors

# 读取 json 文件，把paragraphs字段向量化
with open("TangShi.json", 'r', encoding='utf-8') as file:
	data_list = json.load(file)
	# 提取该json文件中的所有paragraphs字段的值
	text = [data['paragraphs'][0] for data in data_list]

# 向量化文本数据
vectors = vectorize_text(text)

# 将向量添加到原始文本中
for data, vector in zip(data_list, vectors['dense']):
    # 将 NumPy 数组转换为 Python 的普通列表
	data['vector'] = vector.tolist()

# 将更新后的文本内容写入新的json文件
with open("TangShi_vector.json", 'w', encoding='utf-8') as outfile:
	json.dump(data_list, outfile, ensure_ascii=False, indent=4)


# 2 创建集合
# 创建client实例
client = MilvusClient(uri="http://localhost:19530")
# 指定集合名称
collection_name = "TangShi"

# 检查同名集合是否存在，如果存在则删除
if client.has_collection(collection_name):
    print(f"Collection {collection_name} already exists")
    try:
        # 删除同名集合
        client.drop_collection(collection_name)
        print(f"Deleted the collection {collection_name}")
    except Exception as e:
        print(f"Error occurred while dropping collection: {e}")

# 创建集合模式
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
    description="TangShi"
)

# 添加字段到schema
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=256)
schema.add_field(field_name="paragraphs", datatype=DataType.VARCHAR, max_length=10240)
schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=128)

# 创建集合
try:
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        shards_num=2
    )
    print(f"Created collection {collection_name}")
except Exception as e:
    print(f"Error occurred while creating collection: {e}")

# 获取集合的详细信息
collection_info = client.describe_collection(collection_name)
print(f"collection_info: {collection_info}")



# 3 把向量存储到 Milvus 向量数据库
# 读取和处理文件
with open("TangShi_vector.json", 'r') as file:
    data = json.load(file)
    # paragraphs的值是列表，需要获取列表中的字符串，以符合Milvus插入数据的要求
    for item in data:
        item["paragraphs"] = item["paragraphs"][0]

# 将数据插入集合
print(f"正在将数据插入集合：{collection_name}")
res = client.insert(
    collection_name=collection_name,
    data=data
)

# 验证数据是否成功插入集合
print(f"插入的实体数量: {res['insert_count']}")





# 4 创建索引
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
    index_name=vector_index_name
)

print(f"开始创建索引：{vector_index_name}")

# 创建索引
client.create_index(
    # 指定为哪个集合创建索引
    collection_name=collection_name,
    # 使用前面创建的索引参数创建索引
    index_params=index_params
)





# 验证索引
indexes = client.list_indexes(
    collection_name=collection_name  
)
print(f"列出创建的索引：{indexes}")

print("*"*50)

# 查看索引详情
index_details = client.describe_index(
    collection_name=collection_name,  
    # 指定索引名称，这里假设使用第一个索引
    index_name="vector_index"
)

print(f"索引vector_index详情：{index_details}")


# 5 加载集合
print(f"正在加载集合：{collection_name}")
client.load_collection(collection_name=collection_name)

# 验证加载状态
print(client.get_load_state(collection_name=collection_name))




# 6 搜索

# 打印向量搜索结果
def print_vector_results(res):
    """
    打印向量搜索结果。可变位置参数是输出的文本字段，不包括"distance"字段
    """
    # hit是搜索结果中的每一个匹配的实体
    res = [hit["entity"] for hit in res[0]]
    for item in res:
        print(f"title: {item['title']}")
        print(f"author: {item['author']}")
        print(f"paragraphs: {item['paragraphs']}")
        print("-"*50)   
    print(f"数量：{len(res)}")



# 获取查询向量
text = "今天的雨好大"
# text = "我今天好开心"
query_vectors = [vectorize_text([text])['dense'][0].tolist()]

# 设置搜索参数
search_params = {
	# 设置度量类型
	"metric_type": "IP",
	# 指定在搜索过程中要查询的聚类单元数量，增加nprobe值可以提高搜索精度，但会降低搜索速度
	"params": {"nprobe": 16}
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
    output_fields=output_fields
)

print(f"res1:")
print(res1)
print("="*100)
print_vector_results(res1)
print("="*100)



# 搜索案例2，向量搜索，限制distance范围
# 修改搜索参数，设置距离的范围
search_params = {
    "metric_type": "IP",
    "params": {
        "nprobe": 16,
        "radius": 0.2,
        "range_filter": 1.0
    }
}


res2 = client.search(
    collection_name=collection_name,
    # 指定查询向量
    data=query_vectors,
    # 指定搜索的字段
    anns_field="vector",
    # 设置搜索参数
    search_params=search_params,
    # 删除limit参数
    output_fields=output_fields
)

print(f"res2:")
print(res2)
print_vector_results(res2)
print("="*100)




# 搜索案例3，向量搜索+设置过滤条件
filter = f"author == '李白'"

res3 = client.search(
    collection_name=collection_name,
    # 指定查询向量
    data=query_vectors,
    # 指定搜索的字段
    anns_field="vector",
    # 设置搜索参数
    search_params=search_params,
    limit=limit,
    filter=filter,
    output_fields=output_fields
)

print(f"res3:")
print(res3)
print_vector_results(res3)
print("="*100)



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
        print("-"*50)  
    print(f"数量：{len(res)}")



# 构建查询表达式1，包含指定文本
filter = f"paragraphs like '%雨%'"

res4 = client.query(
    collection_name=collection_name,
    filter=filter,
    output_fields=output_fields,
    limit=limit
)


print(f"res4:")
print(res4)
print_scalar_results(res4)
print("="*100)



# 搜索案例5，标量搜索
# 构建查询表达式2，包含指定文本+设置过滤条件
filter = f"author == '李白' && paragraphs like '%雨%'" 

res5 = client.query(
    collection_name=collection_name,
    filter=filter,
    output_fields=output_fields,
    limit=limit
)

print(f"res5:")
print(res5)
print_scalar_results(res5)
print("="*100)