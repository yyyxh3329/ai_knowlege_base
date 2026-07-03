import json
from pathlib import Path
from typing import Any

from app.infra.llm.providers import llm_providers
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state: ImportGraphState):
    # 1、获取相关参数
    chunks = state.get("chunks")
    md_path = state.get("md_path")
    item_name = state.get("item_name")

    # 2、校验
    if not item_name:
        if md_path and Path(md_path).exists():
            item_name = Path(md_path).stem
        if not item_name:
            item_name = "default_item_name"

    if not chunks:
        if md_path:
            chunks_json_obj: Path = Path(md_path).with_name(f"{Path(md_path)}.json")
            chunks = json.load(chunks_json_obj.read_text(encoding="utf-8"))
        if not chunks:
            logger.error(f"chunks为空,读取本地备份文件依然为空,业务无法继续进行!")
            raise ValueError(f"chunks为空,读取本地备份文件依然为空,业务无法继续进行!")

    return chunks, item_name

@step_log("batch_generate_embeddings")
def batch_generate_embeddings(chunks: list[dict[str, Any]], item_name: str, batch_number: int = EMBEDDING_BATCH_SIZE):
    logger.info(f"未生成向量之前的示例:{chunks[0]}")
    length = len(chunks)
    for i in range(0,length, batch_number):
        current_chunks = chunks[i:i + batch_number]
        # 进行拼接
        current_chunks_list = [f"主体:{item_name},内容:{chunk.get('content')}" for chunk in current_chunks]
        # 批量生成
        result = llm_providers.generate_embeddings(current_chunks_list)
        # 回填给当前的chunks current_chunks
        for index, chunk in enumerate(current_chunks):
            chunk["dense_vector"] = result.get("dense")[index]
            chunk["sparse_vector"] = result.get("sparse")[index]
    logger.info(f"生成向量之后的示例:{chunks[0]}")
    return chunks

@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """

    # 1、获取并校验参数
    chunks, item_name = get_data_and_validates(state)
    # 2、批量chunk块 向量的生成 (chunks , item_name , batch_number:int = 6) -> chunks
    embeddings_content = batch_generate_embeddings(chunks, item_name, EMBEDDING_BATCH_SIZE)
    # 3. 更新state embeddings_content
    state['embeddings_content'] = embeddings_content
    return state