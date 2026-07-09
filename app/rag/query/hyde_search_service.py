import logging
import time

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_providers
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state: QueryGraphState):

    item_names = state.get("item_names",[])
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query or len(item_names) == 0:
        logging.error("关联的主体和重写的问题为空，业务无法进行，提前终止！")
        raise ValueError("关联的主体和重写的问题为空，业务无法进行，提前终止！")

    return rewritten_query, item_names

@step_log("call_llm_answer")
def call_llm_answer(rewritten_query:str):

    llm = llm_providers.chat()

    hyde_prompt_text = load_prompt("hyde_prompt",rewritten_query=rewritten_query)

    message = [
        HumanMessage(
            content=hyde_prompt_text,
        )
    ]

    chain = llm | StrOutputParser()

    answer = chain.invoke(message)

    return answer

@step_log("search_by_milvus")
def search_by_milvus(rewritten_query, item_names, llm_answer):
    # 1. rewritten_query进行向量化
    rewritten_query_vectory = llm_providers.generate_embeddings([rewritten_query+":"+llm_answer])
    # 2. 组装reqs请求列表(AnnSearchRequest)
    reqs = milvus_gateway.create_requests(
        dense_vector=rewritten_query_vectory["dense"][0],
        sparse_vector=rewritten_query_vectory["sparse"][0],
        expr=f"item_name in {item_names}",
        limit=5*2
    )
    # 3. 进行混合检索处理
    milvus_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name(),
        reqs=reqs,
        ranker_weights=(0.6,0.4),
        norm_score=True,
        limit=5,
        output_fields=[
            # chunk_id file_title title parent_title part item_name content x x
            #  {} -> lm -> 润色
            "chunk_id",
            "file_title",
            "title",
            "parent_title",
            "part",
            "item_name",
            "content"
        ],
    )
    # milvus_result = [ [id/chunk_id:主键,distance:分,entity:{}] ] -> 外面没有意义! 保证对称性 单列检索
    # 4. 返回结果
    return milvus_result[0] if milvus_result and len(milvus_result) > 0 else []

@step_log("deal_milvus_list")
def deal_milvus_list(llm_milvus_result):
    # {id / chunk_id,distance,entityL{}} -> {}
    hyde_embedding_chunks = []
    for item in llm_milvus_result:
        entity = item.get("entity", {})
        hyde_embedding_chunks.append({
            "id": entity.get("chunk_id"),  # item.get("id") or item.get("chunk_id")
            "score": item.get("distance", 0.0),
            "title": entity.get("title"),
            "file_title": entity.get("file_title"),
            "parent_title": entity.get("parent_title"),
            "part": entity.get("part"),
            "item_name": entity.get("item_name"),
            "content": entity.get("content"),
            "source": "milvus",  # 直接查询 或者假设性查询  milvus 网络检索 web
            "url": ""
        })
    return hyde_embedding_chunks

@step_log("search_by_hyde")
def search_by_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    HyDE 检索服务：
    1. 让 LLM 基于问题虚构一个"理想答案"
    2. 对这个假设性答案进行向量化
    3. 用答案向量在 Milvus 中检索真实文档
    4. 回写 hyde_embedding_chunks
    """
    # 1.获取并校验参数
    rewritten_query, item_names = get_data_and_validates(state)
    # 2. 模型获取答案
    llm_answer = call_llm_answer(rewritten_query)
    # 3. 进行向量的混合+条件检索(item_names rewritten_query) -> [{id/chunk_id:x,distance:0.9,entity:{输出的field} }]
    llm_milvus_result = search_by_milvus(rewritten_query, item_names,llm_answer)
    # 4. 进行结果的统一格式化处理( [{id/chunk_id:x,distance:0.9,entity:{输出的field} }]) -> [{id:x,输出的field,score:,type:milvus,url:""},{},{}]
    hyde_embedding_chunks = deal_milvus_list(llm_milvus_result)
    return hyde_embedding_chunks