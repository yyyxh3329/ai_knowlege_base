import re

from langchain_core.messages import HumanMessage
from app.infra.llm.providers import llm_providers
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task,add_running_task,push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
import time
import sys


#   1. 判断state是否有answer,有我们直接返回(state) -> bool 有没有answer
#            没有 ->  return false
#            有   ->  流式   push_to_session(session_id,delta,{delta:answer })
#                    非流式  return True state => answer
#        2. 上一次返回的值是false
#            3.  answer_prompt_text =  声明一个提示词拼接的方法(state) 拼接
#            4.  调用模型处理字符串answer回答的问题(state,answer_prompt_text)
#                 流式
#                     llm - stream -> push ... -> state answer
#                 非流式
#                     llm - invoke -> state -> answer
#                 state[answer] =
#            5. 提取图片存储到state(reranked [text ![]()] mcp url -> 是不是图片, state)
#                 url -> 后缀名
#                 text -> 网络搜索 md ![xxxx](http://xxx) -> r"\!\[.*?\]\((.*?)\)"  findall
#                 iamge_urls -> state
#        6. 保存聊天记录 [回答]
#        7. 返回state
#        return state

def state_exists_answer(state):
    answer = state.get("answer")
    is_stream = state.get("is_stream")

    # 1. 判断是否存在
    if not answer:
        # 表示answer不存在
        logger.info("此时没有answer数据，表示触发多路召回链路，业务正常流程！跳入回答流程")
        return False

    # 表示answer存在
    if is_stream:
        # 表示流式,将answer放入sse queue队列中
        push_to_session(state.get("session_id"),SSEEvent.DELTA,answer)

    logger.info(f"answer内容不为空,前期没有识别出item_name,提前给与回答!")
    return True


def load_answer_prompt(state):
    # question
    question = state.get("rewritten_query")
    reranked_docs = state.get("reranked_docs",[])

    context = ""
    #  第1部分,标题:xx,来源:网络或者向量库,置信度: xxx,内容: text \n
    for index, doc in enumerate(reranked_docs, start=1):
        context += (f"第{index}部分,标题：{doc.get('title')},来源：{'网络搜索' if doc.get('type') == 'web' else '向量库'},"
                   f"置信度：{doc.get('score')}, 内容：{doc.get('text')} \n")
    # item_names
    item_names = f"{','.join(state.get('item_names',[]))}"
    history_text = ""
    message_list = history_repository.list_recent(state.get("session_id"),limit=6)
    if not message_list or len(message_list) == 0:
        logger.warning(f"当前会话:{state.get('session_id')}没有历史对话记录!提前跳出,history_text为空!")
        history_text = "无对话记录!"
    else:
        final_message_list = [ item for item in message_list if len(item.get("item_names",[])) > 0 ]
        if not final_message_list or len(final_message_list) == 0:
            logger.warning(f"当前会话:{state.get('session_id')}没有有效的历史对话记录!提前跳出,history_text为空!")
            history_text = "无有效对话记录!"
        else:
            for index, item in enumerate(final_message_list, start=1):
                history_text += (f"序号:{index},{'提问:' if item.get('role') == 'user' else '回答:'}"
                                 f"{item.get('rewritten_query') if item.get('role') == 'user' else item.get('text')[:50]},"
                                 f"关联主体: {','.join(item.get('item_names'))} \n")

    # 加载提示词
    answer_out_str = load_prompt("answer_out",history=history_text,item_names=item_names,question=question,context=context)
    return answer_out_str


def call_llm_deal_answer(state, answer_prompt_text):

    # 这里要判断是流式还是非流式，调用模型的方式不一样
    is_stream = state.get("is_stream")
    answer = ""
    llm_client = llm_providers.chat()
    messages = [
        HumanMessage(content=answer_prompt_text)
    ]
    if is_stream:
        # 流式
        stream = llm_client.stream(messages)
        for chunk in stream:
            push_to_session(state.get("session_id"),SSEEvent.DELTA,{"delta":chunk.content})
            answer += chunk.content
    else:
        response = llm_client.invoke(messages)
        answer = response.content

    state["answer"] = answer


def extract_text_image_url(state):
    # 获取reranked_docs  text url
    reranked_docs = state.get("reranked_docs", [])
    # 定义正则规则
    image_re = re.compile(r"\!\[.*?\]\((.*?)\)")  # findall
    image_urls = []
    for doc in reranked_docs:
        # 从向量库中检索出来的图片在文本中
        # 联网搜索中的图片在url中
        url = doc.get("url")
        text = doc.get("text")
        if url and url.endswith(SUPPORTED_IMAGE_EXTENSIONS):
            image_urls.append(url)
        url_list = image_re.findall(text)
        if url_list and len(url_list) > 0:
            image_urls.extend(url_list)

    # image_urls
    state['image_urls'] = image_urls


def save_answer_message_history(state):
    history_repository.save_message(
        session_id=state.get("session_id"),
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", [])
    )


def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + TopK 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """
    ""
    # 1. state中是否存在answer -> 可以返回字符串了
    has_answer: bool = state_exists_answer(state)
    # 2.表示没有answer时，链路走了多路召回，此时要将多路召回的结果 拼接成字符串提示词 交给llm润色给出最终回答
    if not has_answer:
        # answer_prompt_text =  声明一个提示词拼接的方法(state) 拼接
        answer_prompt_text = load_answer_prompt(state)
        # 调用llm模型处理answer字符串的问题
        call_llm_deal_answer(state, answer_prompt_text)
        # 使用正则或者图片url匹配获取image_urls
        extract_text_image_url(state)

    # 3.历史聊天记录记录
    save_answer_message_history(state)
    logger.info(f"终于写完了 2026年6月30日15:56:27")
    return state