import time
from typing import Any

from app.infra.llm.providers import llm_providers
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state:QueryGraphState):
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names",[])

    if not rewritten_query or not item_names:
        logger.error(f"关联的主体或者重写的问题为空,业务无法继续,提前终止!")
        raise ValueError(f"关联的主体或者重写的问题为空,业务无法继续,提前终止!")

    return rewritten_query, item_names

@step_log("search_by_milvus")
def search_by_milvus(rewritten_query:str, item_names:list[str]):
    # 1. rewritten_query进行向量化
    rewritten_query_vectory = llm_providers.generate_embeddings([rewritten_query])
    rewritten_query_dense = rewritten_query_vectory["dense"][0]
    rewritten_query_sparse = rewritten_query_vectory["sparse"][0]
    # 2. 组装reqs请求列表(AnnSearchRequest)
    reqs = milvus_gateway.create_requests(
        dense_vector=rewritten_query_dense,
        sparse_vector=rewritten_query_sparse,
        expr=f"item_name in {item_names}",
        limit=5*2
    )
    # 3.进行混合检索
    milvus_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name(),
        reqs=reqs,
        ranker_weights=(0.6,0.4),
        norm_score=True,
        limit=5,
        output_fields=[
            "chunk_id",
            "file_title",
            "item_name",
            "parent_title",
            "part",
            "content"
        ]
    )
    # 4.返回结果
    return milvus_result[0] if milvus_result and len(milvus_result[0]) > 0 else []

@step_log("deal_milvus_list")
def deal_milvus_list(milvus_result_list:list[dict[str, Any]]):
    embedding_chunks = []
    for item in milvus_result_list:
        # item = {id/chunk_id:x,distance:0.9,entity:{输出的field}
        entity = item.get("entity",{})
        embedding_chunks.append({
            "chunk_id": entity.get("chunk_id"),
            "score": item.get("distance",0.0),
            "title": entity.get("title"),
            "file_title": entity.get("file_title"),
            "parent_title": entity.get("parent_title"),
            "part": entity.get("part"),
            "item_name": entity.get("item_name"),
            "content": entity.get("content"),
            "source":"milvus",
            "url": ""
        })
    return embedding_chunks

@step_log("search_by_embedding")
def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """

    # 1、获取参数并进行校验
    rewritten_query, item_names = get_data_and_validates(state)
    # 2.进行向量的混合+条件检索(item_names rewritten_query)
    milvus_result_list = search_by_milvus(rewritten_query, item_names)
    # 3.进行结果的统一格式化处理 ( [{id/chunk_id:x,distance:0.9,entity:{输出的field} }]) -> [{id:x,输出的field,score:,type:milvus,url:""},{},{}]
    embedding_chunks = deal_milvus_list(milvus_result_list)
    # 4.返回结果
    return embedding_chunks