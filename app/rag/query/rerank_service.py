import time

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_providers
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_SUMMARY_CHAR_RATIO, RERANK_MAX_TOPK, RERANK_MIN_TOPK, \
    RERANK_GAP_ABS, RERANK_GAP_RATIO
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def get_data_and_validates(state:QueryGraphState):
    rewritten_query = state.get("rewritten_query")
    rrf_chunks = state.get("rrf_chunks",[])
    web_search_docs = state.get("web_search_docs",[])

    if not rewritten_query or len(web_search_docs) == 0 or len(rrf_chunks) == 0:
        logger.error(f"rewritten_query,rrf_chunks,web_search_docs可能为空,业务无法继续进行,提前终止!")
        raise ValueError(f"rewritten_query,rrf_chunks,web_search_docs可能为空,业务无法继续进行,提前终止!")

    return rrf_chunks, web_search_docs,rewritten_query


def deal_rrf_and_web_result(rrf_chunks, web_search_docs):
    # 1. 定义一个列表
    reranker_docs = []
    for chunk in rrf_chunks:
        reranker_docs.append({
            "chunk_id":chunk.get("chunk_id"),
            "text": chunk.get("content"),
            "score":chunk.get("score"),
            "title":chunk.get("title"),
            "type":"milvus",
            "url": None
        })
    for doc in web_search_docs:
        reranker_docs.append({
            "chunk_id": None,
            "text": doc.get("snippet"),
            "title": doc.get("title"),
            "score": 0,  # rrf分 -> reranker打的分
            "type": "web",
            "url": doc.get("url")
        })

    return reranker_docs


def create_question_answer_list(rewritten_query, reranker_docs):
    question_answer_pair_list = []
    # 因为reranker模型的维度为 512 也就是我的问题和回答的长度不能超过512
    # 在此处需要对 rrf融合后的chunk中的text和联网搜索中的结果进行处理，如果超过512，就进行精简处理
    # 1. 获取rewritten_query并且计算token数量
    reranker_mode = llm_providers.reranker_mode()
    tokenizer = reranker_mode.tokenizer
    # 算的时候,只需要算我这个字符占有token列表,不用考虑前后的特殊标识
    rewritten_query_tokens_list = tokenizer.encode(rewritten_query,add_special_tokens=False)
    rewritten_query_token_len = len(rewritten_query_tokens_list)
    # 2、 遍历rewritten_query问题的每个答案(包括两路检索及联网搜索)
    for doc in reranker_docs:
        # 3. 答案的长度判读
        answer = doc.get("text")
        answer_list = tokenizer.encode(answer,add_special_tokens=False)
        answer_len = len(answer_list)
        # 4. 超长了调用模型进行压缩
        # reranker固定4个分割符号
        if answer_len + rewritten_query_token_len + 4 > RERANK_MAX_INPUT_TOKENS:
            # 调用模型进行压缩
            # limit = 答案的token / 1.3 -> int -> 50 max
            # 此时万一问题很长，导致回答能占的token数很小，也会给到一个最小值RERANK_SUMMARY_CHAR_RATIO = 50
            limit = max((RERANK_MAX_INPUT_TOKENS - 4 - rewritten_query_token_len) / RERANK_SUMMARY_CHAR_RATIO,
                RERANK_SUMMARY_CHAR_RATIO)
            # 加载提示词
            rerank_text_refine_str = load_prompt("rerank_text_refine",question=rewritten_query,answer=answer,limit=limit)
            message = [
                HumanMessage(content=rerank_text_refine_str),
            ]
            llm = llm_providers.chat()
            chain = llm | StrOutputParser()
            answer = chain.invoke(message)
        # 5. 答案一定处理过了
        # question_answer_pair_list answer -> 可能被压缩 -> 只用于打分
        question_answer_pair_list.append([rewritten_query,answer])
    # 6. 返回结果
    return question_answer_pair_list


def use_reranker_deal_score(question_answer_pair_list, reranker_docs):

    reranker_mode = llm_providers.reranker_mode()
    # 这里会按顺序获取分数列表
    scores_list = reranker_mode.compute_score(question_answer_pair_list,normalize=True)
    # 将分数按按顺序补充到reranker_docs中
    for score,doc in zip(scores_list, reranker_docs):
        doc["score"] = score
    # 给reranker_docs做最终的排序
    reranker_docs.sort(key=lambda x: x.get("score",0.0),reverse=True)


def dyn_limit_reranker_docs(reranker_docs):
    """
      动态结果截取! topk个
         RERANK_MAX_TOPK: int = 5  -> 最多10个
         RERANK_MIN_TOPK: int = 2  -> 最少2个
         RERANK_GAP_RATIO: float = 20% -> 断崖百分比  ->  1 - 2 / 1     0.3 0.2 -> (0.3 - 0.2) / 0.3 = 33%
         RERANK_GAP_ABS: float = 0.2   -> 断崖分差值  ->  0.8  ->  0.5  跳过  大 多了 影响准确了  小  少了 召回率
      topk -> ???
    :param reranker_docs:
    :return:
    """
    top_max: int = RERANK_MAX_TOPK
    top_min: int = RERANK_MIN_TOPK
    gap_abs: float = RERANK_GAP_ABS  # 0.2
    gap_ratio: float = RERANK_GAP_RATIO  # 0.2
    # 情况1：如果查询到doc的数量小于top_max，这里做一个容错将doc的数量赋值给top_max
    top_max = min(top_max, len(reranker_docs))
    # 情况2：此时的top_max可能会小于top_min，因为top_max被赋值为len(reranker_docs)了，这时候要做判断if top_max > top_min
    # 情况3：topk没有被赋值：1.top_max > top_min  2.就没有断崖  这两种情况怎么办？
    topk:int= top_max # 这个可以处理1.top_max > top_min  2.就没有断崖
    if top_max > top_min:
        for pre_index in range(top_min-1, top_max-1):
            # 获取pre_index对象的前置分数
            pre_score = reranker_docs[pre_index].get("score",0.0)
            # 获取next_index对应后置分数
            next_score = reranker_docs[pre_index+1].get("score",0.0)

            # 分差
            abs_score = pre_score - next_score
            ratio = abs_score / pre_score

            if abs_score > gap_abs or ratio > gap_ratio:
                # 表示出现断崖
                topk = pre_index + 1
                break

    reranker_docs = reranker_docs[:topk]
    return reranker_docs

def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """
    # 1.获取并且校验参数(state) rewritten_query  rrf_chunks  web_search_docs，因为后面需要调用reranker模型，传入的数据中需要rewritten_query
    rrf_chunks, web_search_docs,rewritten_query = get_data_and_validates(state)
    # 2. 数据格式化处理，将联网索搜索和rrf_chunks统一格式
    reranker_docs = deal_rrf_and_web_result(rrf_chunks,web_search_docs)
    # 3. 组装问题和答案的列表(rewritten_query,reranker_list) -> question_answer_pair_list [[问题,回答],[],[]]
    question_answer_pair_list = create_question_answer_list(rewritten_query, reranker_docs)
    # 4、进行打分和排序
    logger.info(f"排序和打分之前的数据:{reranker_docs}")
    use_reranker_deal_score(question_answer_pair_list, reranker_docs)
    logger.info(f"排序和打分之后的数据:{reranker_docs}")
    # 5.动态截取数据
    reranker_docs = dyn_limit_reranker_docs(reranker_docs)
    # 6.更新state
    state["reranked_docs"] = reranker_docs
    return state