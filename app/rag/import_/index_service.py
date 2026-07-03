from typing import Any

from pymilvus import DataType

from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log

@step_log("require_embeddings_content")
def require_embeddings_content(state:ImportGraphState):
    embeddings_content = state.get("embeddings_content",[])

    # 如果 embeddings_content 为空，无法继续业务，直接抛出异常终止流程
    if not embeddings_content:
        logger.error("embeddings_content为空,无法继续业务!!")
        raise ValueError("embeddings_content为空,无法继续业务!!")

    return embeddings_content

@step_log("prepare_chunks_collection")
def prepare_chunks_collection():
    # 获取 Milvus 客户端
    milvus_client = milvus_gateway.milvus_client()
    # 获取集合名称（从配置中读取）
    collection_name = milvus_gateway.chunk_collection_name()
    # 如果集合已存在，直接返回，无需重复创建
    if milvus_client.has_collection(collection_name=collection_name):
        return

    # 不存在就创建
    # 创建 schema，启用自动 ID 和动态字段
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    # 添加主键字段：chunk_id，INT64 类型，自增
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    # 添加文件标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    # 添加主体名称字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    # 添加切片标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    # 添加父标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
    # 添加切片序号字段：INT8 类型
    schema.add_field(field_name="part", datatype=DataType.INT8)
    # 添加内容字段：VARCHAR 类型，最大长度 65535（支持长文本）
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    # 添加稠密向量字段：FLOAT_VECTOR 类型，维度 1024
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR 类型
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 准备索引参数
    index_params = milvus_client.prepare_index_params()

    # 为稠密向量创建索引：使用 AUTOINDEX，metric_type 为 IP（内积）
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={
            "M": 64,  # 每个点最大的链接数量
            "efConstruction": 100  # 考虑链接的范围 在这个范围内确定 M
        }
    )

    # 为稀疏向量创建索引：使用 SPARSE_INVERTED_INDEX，算法为 DAAT_MAXSCORE
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    # 创建集合并应用索引
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

@step_log("remove_old_chunks")
def remove_old_chunks(file_title):
    milvus_client = milvus_gateway.milvus_client()

    milvus_client.delete(
        collection_name=milvus_gateway.chunk_collection_name(),
        filter=f"file_title == '{file_title}'"
    )

@step_log("insert_embeddings_content")
def insert_embeddings_content(embeddings_content:list[dict[str,Any]]):
    milvus_client = milvus_gateway.milvus_client()

    result = milvus_client.insert(
        collection_name=milvus_gateway.chunk_collection_name(),
        data=embeddings_content,
    )

    # 记录插入结果
    logger.info(f"插入数据成功! 总条数:{result.get('insert_count', 0)}")
    logger.info(f"插入数据主键回显:{result.get('ids', [])}")

@step_log("index_chunks")
def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    入库服务：
    1. 准备集合 schema 和索引
    2. 根据 item_name 删除旧数据
    3. 批量插入新的 chunks
    4. 回写 chunk_id 等入库结果
    """
    # 1.先校验切片存在，避免把空数据写入向量库
    embeddings_content = require_embeddings_content(state)
    # 2.集合不存在时先自动创建，保证首次导入也能直接跑通
    prepare_chunks_collection()
    # 3.获取文件名称，根据文件名称删除之前存入的该文件的向量块,然后在进行导入
    file_title = state.get("file_title", "")
    if file_title:
        remove_old_chunks(file_title)
    # 4.批量插入数据到向量数据库
    insert_embeddings_content(embeddings_content)
    return state