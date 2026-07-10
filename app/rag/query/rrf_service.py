import time

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import NODE_RRF_K, NODE_RRF_LIMIT_TOP
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state:QueryGraphState):
    embedding_chunks = state.get("embedding_chunks",[])
    hyde_embedding_chunks= state.get("hyde_embedding_chunks",[])

    if  len(hyde_embedding_chunks)==0 or len(embedding_chunks)==0:
        logger.error(f"embedding_chunks或者hyde_embedding_chunks数据为空,业务无法继续进行,提前终止!")
        raise ValueError(f"embedding_chunks或者hyde_embedding_chunks数据为空,业务无法继续进行,提前终止!")

    return embedding_chunks, hyde_embedding_chunks

@step_log("use_by_rrf")
def use_by_rrf(rrf_list:list,k:int=NODE_RRF_K,top:int=NODE_RRF_LIMIT_TOP):
    score_dict: dict[str, float] = {}  # chunk_id , 0.56
    chunk_dict: dict[str, dict] = {} # 这里新建一个字典，用来将两路的chunk块去重后存在一起，方便后面取
    # 这两路的chunk_id 有些是相同的有些是不同的
    # 外面循环遍历的是两路：向量检索和假设下向量检索，这样就可以将所有的chunk_id的分数全都添加到score_dict中
    for weight,current_chunks_list in rrf_list:
        for rank,chunk in enumerate(current_chunks_list,start=1):
            # 此时的rank就是分数排名的顺序
            # chunk_id 的rrf后的分数 = 上一次计算的分数 + weight * 1 / (k + rank)
            score_dict[chunk.get("chunk_id")] = score_dict.get(chunk.get("chunk_id"),0.0) + weight * 1 / (rank + k)
            chunk_dict.setdefault(chunk.get("chunk_id"),chunk)

    chunk_list = []
    # 遍历已经算好分 score_dict
    for chunk_id,score in score_dict.items():
        chunk = chunk_dict.get(chunk_id,{})
        chunk["score"] = score
        chunk_list.append(chunk)
    # 给chunk_list排序
    chunk_list.sort(key=lambda x:x.get("score",0),reverse=True)
    # 截取最高分数的chunk
    return chunk_list[:top]

@step_log("fuse_by_rrf")
def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档列表（Top 10）
    4. 回写 rrf_chunks
    """
    # 1、获取参数并校验
    embedding_chunks, hyde_embedding_chunks = get_data_and_validates(state)
    # 2、目标方便遍历和获取对应的权重
    rrf_list = [(1.0,embedding_chunks),(1.0,hyde_embedding_chunks)]
    # 3. 使用rrf算法进行数据处理( list) -> rrf_chunks[5]
    rrf_chunks = use_by_rrf(rrf_list)
    # 4. 修改state = rrf_chunks
    state['rrf_chunks'] = rrf_chunks
    # 5. 返回state
    return state