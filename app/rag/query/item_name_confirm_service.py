from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_providers
from app.infra.persistence.history_repository import history_repository
from app.infra.vector_store.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state: QueryGraphState) -> tuple[str,str]:
    # 1.获取参数
    session_id = state.get("session_id")
    original_query = state.get("original_query")
    # 2.校验
    if not session_id or not original_query:
        logger.error(f"session_id或者original_query为空,业务无法继续进行,提前终止!")
        raise ValueError(f"session_id或者original_query为空,业务无法继续进行,提前终止!")

    return session_id, original_query

@step_log("get_history_messages_and_context")
def get_history_messages_and_context(session_id:str):
    # 1.先进行查询
    message_list:list[dict] = history_repository.list_recent(session_id=session_id,limit=10)
    # 2.校验有没有查询到
    if not message_list or len(message_list) == 0:
        logger.warning(f"当前会话:{session_id}没有历史对话记录!提前跳出,history_text为空!")
        return "无对话记录!"
    # 3.有聊天记录，需要做有效判断
    final_message_list = [
        item for item in message_list if len(item.get("item_names",[])) > 0
    ]
    if not final_message_list or len(final_message_list) == 0:
        logger.warning(f"当前会话:{session_id}没有有效的历史对话记录!提前跳出,history_text为空!")
        return "无有效对话记录!"
    # 4. 拼接history_text
    #  序号:1,提问:重写的问题,关联主体:1,2,3,4
    #  序号:2,回答:回答内容[:50],关联主体:1,2,3,4
    history_text = ""
    for index, item in enumerate(final_message_list,start=1):
        history_text += (f"序号{index}, {'提问' if item.get('role') == 'user' else '回答'} "
                         f"{ item.get('rewritten_query') if item.get('role') == 'user' else item.get('text')[:50] },"
                         f"关联主体：{','.join(item.get('item_names',[]))} \n")
    return history_text

@step_log("call_llm_item_name_and_rewritten")
def call_llm_item_name_and_rewritten(history_text, original_query):
    # 1、加载模型对象
    llm_client = llm_providers.chat()
    # 2、加载提示词
    # 提示词 1. history靠上! 影响了模型对规则读取 历史对话向下挪
    #       2. 不是每次提问都是延续的! 上一次 烫金机  本次  苹果手机
    history_prompt_text = load_prompt("rewritten_query_and_itemnames",history_text=history_text,query=original_query)
    # 3、包装提示词
    message = [
        HumanMessage(
            content=history_prompt_text,
        )
    ]
    chain = llm_client | JsonOutputParser()
    result_dict = chain.invoke(message)
    if "item_names" not in result_dict:
        result_dict["item_names"] = []
    if "rewritten_query" not in result_dict:
        result_dict["rewritten_query"] = original_query
    return result_dict

@step_log("search_by_item_names")
def search_by_item_names(item_names:list[str]):
    # 准备一个最终的字典
    final_result = {}
    # 1.先将传过来的item_names向量化
    item_names_vectors:dict[str,list] = llm_providers.generate_embeddings(item_names)
    # 2.通过item_names的长度遍历次数
    for index in range(0,len(item_names)):
        # 3.获取稠密和稀疏向量
        item_name = item_names[index]
        item_name_dense = item_names_vectors["dense"][index]
        item_name_sparse = item_names_vectors["sparse"][index]
        # 3.1 创建annSearchRequest
        req_list = milvus_gateway.create_requests(item_name_dense, item_name_sparse,limit=5*2)
        # 3.2 创建WeightReranker排序器
        # 3.3 进行混合检索
        result = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_name_collection_name(),
            reqs=req_list,
            ranker_weights=(0.5,0.5),
            norm_score=True,
            output_fields=["item_name"]
        )
        # result =>  [[ {id:主键1,distance:0.9,entity:{item_name1:具体的name}},{id:主键2,distance:0.9,entity:{item_name2:具体的name}} ]]
        # 4.处理检索后的结果
        item_name_milvus_list = []
        if len(result[0]) > 0:
            for item in result[0]:
            # {id:主键,distance:0.9,entity:{item_name:具体的name}}
                item_name_milvus_list.append({
                    "item_name": item.get('entity').get('item_name'), "score": item.get('distance')
                })
        # 5.循环完以后得结果最终返回即可
        final_result[item_name] = item_name_milvus_list
    return final_result

@step_log("select_item_names_by_source")
def select_item_names_by_source(milvus_result:dict[str,list[dict]]):

    confirmed_list = []
    option_list = []

    for item_name,milvus_result_list_dict  in milvus_result.items():
        # milvus_result_list_dict = [{item_name:"向量数据库中的item_name",score:0.9 }, 0.85 0.8 ..]
        # [{item_name:"向量数据库中的item_name",score:0.8},..] 数据是已经排好顺序的! 分高 前面!
        # 如何根据分数 定义是可以确认和不完全确认的主体？  也就是分值界定的标准
        # confirmed_list = [] -> 啥样的算确认  [ 稠密向量满分 1 * 0.5 + 稀疏向量满分 0.75  0.5 ] = 0.5 + 0.375 = 0.875 -> 0.8 + 确认
        # option_list = [] -> 算可选的 -> 0.6 - 0.8 -> 可选
        high_score_list = [item for item in milvus_result_list_dict if item.get('score') >= 0.70]
        md_score_list = [ item for item in milvus_result_list_dict if 0.6 <= item.get('score') < 0.70]


        if len(high_score_list) > 0:
            confirmed_list.append(high_score_list[0])
            logger.info(f"模型识别item_name:{item_name},有对应向量数据库中确认的item_name:{high_score_list[0].get('item_name')}")
            continue

        if len(md_score_list) > 0:
            option_list.extend(md_score_list[:2])
            logger.info(
                f"模型识别item_name:{item_name},没有对应向量数据库中确认的item_name,"
                f"但是有可选的:{','.join([ item.get('item_name') for item in md_score_list[:2]])}")
            continue

    return {
        "confirmed_list": confirmed_list,
        "option_list": option_list
    }

@step_log("apply_item_name_result")
def apply_item_name_result(state:QueryGraphState, list_dict:dict, rewritten_query:str):
    """
          本次就是为了state
             confirmed_list ->  有数据
               item_names = []
               rewritten_query = ""
               一定不能给 answer del ...
               return
             option_list ->confirmed_list没有,  有数据
               item_names 不赋值
               rewritten_query 也无需赋值
               answer = 本次问题没有识别到关联的主体,可能是: 1,2,3,4 请您确认和选择!
               return
             confirmed_list,option_list -> 都没有数据
               item_names 不赋值
               rewritten_query 也无需赋值
               answer = 本次问题没有关联到任何主体,有没有相似可选的主体! 请您明确主体再提问!
               return
        :param state:
        :param list_dict:
        :param rewritten_query:
        :return:
        """
    confirmed_list = list_dict.get("confirmed_list",[])
    option_list = list_dict.pop("option_list",[])

    if len(confirmed_list) > 0:
        state['item_names'] = [item.get('item_name') for item in confirmed_list]
        state['rewritten_query'] = rewritten_query
        if "answer" in state:
            state["answer"] = None
        return

    if len(option_list) > 0:
        # 没有确认,但是有可选的
        # option_list = [[],[]]
        state['answer'] = f"本次提问没有确认主体,但是有相似可选的:{','.join([ item.get('item_name') for item in option_list ])},请您再次确认!"
        return

    state['answer'] = "本次问题没有关联到任何主体,有没有相似可选的主体! 请您明确主体再提问!"

@step_log("save_history_message")
def save_history_message(state:QueryGraphState):
    history_repository.save_message(
        session_id=state.get('session_id'),
        role="user",
        text=state.get("original_query"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names",[]),
        image_urls=[]
    )


def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取商品名
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动对齐标准型号，或生成反问让用户手动确认
    5. 同步历史记录到 MongoDB
    """
    # 1.校验参数
    session_id,original_query = get_data_and_validates(state)
    # 2.获取(有效的)历史聊天记录，并且情节成history_test提示词
    history_text = get_history_messages_and_context(session_id)
    # 3. 调用模型识别item_names和重写的问题
    result_dict:dict = call_llm_item_name_and_rewritten(history_text,original_query)
    list_dict = {}
    # 4.进行向量数据库的搜索
    if len(result_dict.get("item_names",[])) > 0:
        milvus_result:dict[str,list[dict]] = search_by_item_names(result_dict.get("item_names",[]))
        # 5. 根据打分确定两个列表 确认列表 可选列表
        # dict{"confirmed_list":[],option_list:[]}
        #   "confirmed_list":confirmed_list,
        #   "option_list":option_list
        list_dict: dict[str, list] = select_item_names_by_source(milvus_result)
    # 6. 确定和可选列表修改state answer item_names rewritten_query
    # 修改 state answer item_names rewritten_query list_dict
    apply_item_name_result(state,list_dict,result_dict.get("rewritten_query"))
    # 7.保存此次聊天对话记录
    save_history_message(state)
    return state