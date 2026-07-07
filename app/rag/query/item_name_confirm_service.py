from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_providers
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def get_data_and_validates(state: QueryGraphState) -> tuple[str,str]:
    # 1.获取参数
    session_id = state.get("session_id")
    original_query = state.get("original_query")
    # 2.校验
    if not session_id or not original_query:
        logger.error(f"session_id或者original_query为空,业务无法继续进行,提前终止!")
        raise ValueError(f"session_id或者original_query为空,业务无法继续进行,提前终止!")

    return session_id, original_query


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


def call_llm_item_name_and_rewritten(history_text, original_query):
    # 1、加载模型对象
    llm_client = llm_providers.chat()
    # 2、加载提示词
    # 提示词 1. history靠上! 影响了模型对规则读取 历史对话向下挪
    #       2. 不是每次提问都是延续的! 上一次 烫金机  本次  苹果手机
    history_prompt_text = load_prompt("rewritten_query_and_itemnames.prompt",history_text=history_text,query=original_query)
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

    return state