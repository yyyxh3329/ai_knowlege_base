import time

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.infra.persistence.history_repository import history_repository
from app.infra.llm.providers import llm_providers

#   1.获取并且校验参数(state) -> session_id / original_query
#         session_id,original_query = get_data_and_validates(state)
#          获取数据
#          非空校验
#          返回结果
#       2.获取(有效的)历史聊天记录,并且拼接成history_text提示词
#          get_history_messages_and_context(session_id) -> history_text
#          获取mongodbclient
#          根据session_id获取最近聊天录(10)
#          列表推导式推到有效的messages [item_names]
#          将有效的messages拼接成history_text
#             {id session_id  序号:1 / 2    role [提问还是回答] text [user ->原始问题 -> rewritten_query assistant -> 回答 -> text [:60]]
#             rewritten_query item_names => [主体: x,x,x]  image_urls ts} -> item_name识别和问题重写
#          return history_text
#       3.调用模型识别item_names和rewritten_query -> json [item_names不一定准 大模型]
#          dict = call_llm_item_name_and_rewritten(history_text,original_query)
#          获取模型对象
#          加载提示词字符串
#          封装提示词对象
#          包装chains -> StrOutput... JsonOutput ->
#          dict = 执行练获取字典   面试题: 怎么确保大语言模型返回的数据是JSON格式!! 1. 提示词强调返回数据格式提供返回数据示例
#                                                                          2. 模型的json模式设置
#                                                                          3. 做好返回值的参数校验 JsonOutputParser ``` json  json dict
#                                                                          4. 返回字典key的校验 item_names -> 无 -> []
#                                                                                             rewritten_query -> 无 -> original_query
#          return dict


def get_data_and_validates(state:QueryGraphState) -> tuple[str,str]:
    """
    参数校验
    :param state:
    :return:
    """
    # 1. 获取数据
    session_id = state.get("session_id")
    original_query = state.get("original_query")
    # 2. 进行数据校验
    if not session_id or not original_query:
        logger.error(f"session_id或者original_query未空,业务无法继续进行,提前终止!")
        raise ValueError(f"session_id或者original_query未空,业务无法继续进行,提前终止!")
    # 3. 返回结果
    return session_id,original_query


def get_history_messages_and_context(session_id:str) -> str:
    """
     获取近期有效的历史聊天记录并且拼接成上下文
    :param session_id:
    :return:
    """
    # 1.获取最近10条历史条件记录
    message_list:list[dict] =  history_repository.list_recent(session_id=session_id,limit=10)
    if not message_list or len(message_list) == 0:
        logger.warning(f"当前会话:{session_id}没有历史对话记录!提前跳出,history_text为空!")
        return "无对话记录!"
    # 2.有聊天记录,需要做有效判断
    final_message_list = [
        item
        for item in message_list if len(item.get("item_names",[])) > 0
    ]
    if not final_message_list or len(final_message_list) == 0:
        logger.warning(f"当前会话:{session_id}没有有效的历史对话记录!提前跳出,history_text为空!")
        return "无有效对话记录!"
    # 3. 拼接history_text
    #     {id session_id  序号:1 / 2    role [提问还是回答] text [user ->原始问题 -> rewritten_query assistant -> 回答 -> text [:60]]
    # #             rewritten_query item_names => [主体: x,x,x]  image_urls ts} -> item_name识别和问题重写
    history_text = ""
    for index, item in  enumerate(final_message_list,start=1):
        history_text += (f"序号:{index},{'提问:' if item.get('role') == 'user' else '回答:'}"
                         f"{item.get('rewritten_query') if item.get('role') =='user' else item.get('text')[:50]},"
                         f"关联主体: {','.join(item.get('item_names'))} \n")
        #  序号:1,提问:重写的问题,关联主体:1,2,3,4
        #  序号:2,回答:回答内容[:50],关联主体:1,2,3,4
    return history_text


def call_llm_item_name_and_rewritten(history_text:str, original_query:str)->dict:
    """
    调用模型进行识别
    :param history_text: 
    :param original_query: 
    :return: 
    """
    #1. 加载模型对象
    json_llm_client = llm_providers.chat(json_mode=True)
    #2. 加载提示词
    # 提示词修改
    # 提示词 1. history靠上! 影响了模型对规则读取 历史对话向下挪
    #       2. 不是每次提问都是延续的! 上一次 烫金机  本次  苹果手机
    history_prompt_text = load_prompt("rewritten_query_and_itemnames",query=original_query,history_text=history_text)
    #3. 包装提示词对象
    history_prompt_messages = [
        HumanMessage(
            content=history_prompt_text
        )
    ]
    #4. 创建调用链
    chains = json_llm_client | JsonOutputParser()
    #5. 调用获取结果
    result_dict = chains.invoke(history_prompt_messages)
    #6. 参数校验赋予默认值
    if "item_names" not in result_dict:
        result_dict["item_names"] = []
    if "rewritten_query" not in result_dict:
        result_dict["rewritten_query"] = original_query
    #7. 返回结果
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
    # 1
    session_id, original_query = get_data_and_validates(state)
    # 2 获取历史(有效)信息和拼接上下文
    history_text = get_history_messages_and_context(session_id)
    # 3. 调用模型识别item_names和重写的问题
    result:dict =call_llm_item_name_and_rewritten(history_text,original_query)
    return state