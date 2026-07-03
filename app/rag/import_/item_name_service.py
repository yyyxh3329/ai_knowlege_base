import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.infra.llm.providers import llm_providers
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNKS_SPLIT_TOP_NUMBER
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state) -> tuple[list[dict[str, Any]], str]:
    md_path = state.get("md_path")
    file_title = state.get("file_title")
    chunks = state.get("chunks")

    if not file_title:
        if md_path and Path(md_path).exists():
            file_title = md_path.name
        else:
            file_title = "default_name"
        logger.warning(f"file_title没有值，给予默认值{file_title}")

    if not chunks:
        if md_path:
            chunks_json_obj: Path = Path(md_path).with_name(f"{Path(md_path).stem}.json")
            chunks = json.loads(chunks_json_obj.read_text(encoding="utf-8"))
        if not chunks:
            logger.error(f"chunks为空,读取本地备份文件依然为空,业务无法继续进行!")
            raise ValueError(f"chunks为空,读取本地备份文件依然为空,业务无法继续进行!")

    return chunks, file_title

@step_log("recognize_item_name_by_chunks")
def recognize_item_name_by_chunks(chunks: list[dict[str, Any], str], file_title: str):
    # 1.获取语言模型对象
    llm = llm_providers.chat()
    # 2.加载提示词
    system_prompt_text: str = load_prompt("product_recognition_system")
    # 切去部分chunk来进行识别
    used_chunks = chunks[:CHUNKS_SPLIT_TOP_NUMBER]
    # 将每个chunk拼接在一起，组成一个use_content
    use_content: str = ""
    for index, chunk in enumerate(used_chunks, start=1):
        use_content += f"第{index}部分: 标题：{chunk.get('title')} ，内容为{chunk.get('content')} \n"
    user_prompt_text: str = load_prompt("item_name_recognition", file_title=file_title, context=use_content)
    # 3.封装成Message
    message = [
        SystemMessage(content=system_prompt_text),
        HumanMessage(content=user_prompt_text)
    ]
    chain = llm | StrOutputParser()
    item_name = chain.invoke(message)

    if not item_name:
        item_name = file_title
        logger.warning(f"模型未识别到item_name使用file_title:{file_title}赋予默认值!")
    return item_name

@step_log("chunk_update_item_name")
def chunk_update_item_name(chunks, item_name):
    for chunk in chunks:
        chunk["item_name"] = item_name

    logger.info(f"完成chunk的item_name属性更新: {item_name}")

@step_log("prepared_milvus_item_name_collection")
def prepared_milvus_item_name_collection():
    # 1.获取milvus客户端
    milvus_client = milvus_gateway.milvus_client()
    # 2.获取要写入的集合名
    item_name_collection_name = milvus_gateway.item_name_collection_name()
    # 3.查看向量数据库是否有这个集合名，没有就创建
    collection_is_exist = milvus_client.has_collection(collection_name=item_name_collection_name)
    if collection_is_exist:
        logger.info(f"向量数据库中存在{item_name_collection_name},无需创建！")
        # 直接返回
        return

    # 4.不存在的就创建集合
    logger.info(f"{item_name_collection_name}集合不存在,开始创建!")
    # 4.1创建schema
    schema = milvus_client.create_schema(
        auto_id=True, # 主键自增长
        enable_dynamic_field=True, # schema设置列的信息! True插入了没有提前设定好的列,也可以进行存储
    )
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 4.2创建索引
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE", # 稠密向量相似度可以选: COSINE = IP -> 速度更快一些 归一化   L2
        params={
            "M": 64, # 每个点最大的链接数量
            "efConstruction": 100 # 考虑链接的范围 在这个范围内确定 M
        }
    )

    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX", # 倒排索引
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}  # 根据权重值做优化,降低一些低权重数据的排名!!!
    )
    # 4.3 创建集合
    milvus_client.create_collection(
        collection_name=item_name_collection_name,
        schema=schema,
        index_params=index_params,
    )

@step_log("delete_and_insert_item_name")
def delete_and_insert_item_name(item_name, file_title):
    # 1.根据item_name生成稠密稀疏向量
    result = llm_providers.generate_embeddings([item_name])
    dense_vector = result.get("dense")[0]
    sparse_vector = result.get("sparse")[0]
    # 2.先对向量数据库中文件名file_title对应的整条数据进行删除，只有一条
    client = milvus_gateway.milvus_client()
    # 这是一个幂等操作
    client.delete(
        collection_name=milvus_gateway.item_name_collection_name(),
        filter=f"file_title == '{file_title}'",
        # filter=f"file_title == 'hk180烫金机安全手册'"
    )

    data = [
        {
            "file_title": file_title,
            "item_name": item_name,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }
    ]
    # 3.数据进行插入
    client.insert(
        collection_name=milvus_gateway.item_name_collection_name(),
        data=data
    )
    logger.info(f"完成{item_name}的数据更新或者插入!")

@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """

    # 1.获取并且校验参数(state)
    chunks, file_title = get_data_and_validates(state)
    # 2.根据chunks内容让大语言模型识别item_name，如果识别不出来，使用file_title设为item_name
    item_name = recognize_item_name_by_chunks(chunks, file_title)
    # 3.给chunk块补全item_name内容
    chunk_update_item_name(chunks, item_name)
    # 4. 提前在准备对应的milvus集合数据
    prepared_milvus_item_name_collection()
    # 5.插入item_name数据
    delete_and_insert_item_name(item_name,file_title)
    # 6.更新state chunks
    state["chunks"] = chunks
    state["item_name"] = item_name
    return state
