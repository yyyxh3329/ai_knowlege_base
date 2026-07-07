from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi.responses import FileResponse,StreamingResponse
from fastapi import FastAPI, BackgroundTasks,Request
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.query_schema import HealthResponseSchema, QueryRequestSchema, QueryStreamResponseSchema, \
    QueryNotStreamResponseSchema, HistoryListResponseSchema, HistoryItemResponseSchema
from app.infra.persistence.history_repository import history_repository
from app.process.import_.agent.state import create_default_state
from app.process.query.agent.mian_graph import query_app
from app.process.query.agent.state import QueryGraphState
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.shared.utils import task_utils
from app.shared.utils.sse_utils import create_sse_queue, push_to_session, SSEEvent, sse_generator
from app.shared.utils.task_utils import get_done_task_list, TASK_STATUS_PROCESSING, clear_task, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED

# 定义fastApi对象

app = FastAPI(
    title=settings.query_app_name,
    description="描述,进行rag查询的服务对象",
    version="0.2.0"
)

# 跨域处理 [前端和后端可能不在一个服务器,非同源,浏览器认为跨域,拦截了 ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)


# 1.接口一  返回页面
@app.get("/html")
def query_html():
    html_path_obj: Path = PROJECT_ROOT / "app" / "resources" / "html" / "chat.html"

    return FileResponse(
        path=str(html_path_obj),
        media_type=guess_type(html_path_obj.name)[0],
    )


# 接口2: /health
# {code:200,message=xx}
@app.get("/health")
def health():
    logger.info(f"完成健康检查!{datetime.now()}")
    return HealthResponseSchema(
        code=200,
        message=f"完成健康检查!{datetime.now()}"
    )


def invoke_query_graph(session_id: str, original_query: str, is_stream: bool) -> QueryGraphState:
    try:
        # 任务整体状态监控!!

        # 这里要情清空task字典，不管是流式还是非流式都需要
        clear_task(session_id)
        # 如果是流式，这里创建一个新的队列
        if is_stream:
            create_sse_queue(session_id)

        # 更新任务的执行状态为  processing，此时为流式会将这个session_id的 1任务状态 2已完成任务节点列表 3正在运行的节点列表 添加到队列中
        task_utils.update_task_status(session_id, TASK_STATUS_PROCESSING, push_queue=is_stream)
        # 封装成state
        state = create_default_state(
            session_id=session_id,
            original_query=original_query,
            is_stream=is_stream
        )
        # 执行图对象
        result_state = query_app.invoke(state)
        # 更新任务的执行状态为  processing
        task_utils.update_task_status(session_id, TASK_STATUS_COMPLETED, push_queue=is_stream)

        # 执行完毕后，[final - sse - is_stream = True]
        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": result_state.get('answer'),
                    "status": "completed",
                    "image_urls": result_state.get('image_urls', [])
                }
            )

    except Exception as e:
        task_utils.update_task_status(session_id, TASK_STATUS_FAILED, push_queue=is_stream)
        logger.exception(f"执行查询流程报错,错误信息:{str(e)}")


# 查询接口
@app.post("/query")
def query(backgroundtasks: BackgroundTasks, query_params: QueryRequestSchema):
    # 1、获取参数
    query = query_params.query
    session_id = query_params.session_id
    is_stream = query_params.is_stream
    # 2、封装执行图方法
    # 3、判断是否为流式
    if is_stream:
        # 流式 异步的执行图
        backgroundtasks.add_task(invoke_query_graph, session_id=session_id, original_query=query, is_stream=is_stream)
        # 此时立即返回响应
        return QueryStreamResponseSchema(
            session_id=session_id,
            message=f"已经开始了问题：{query}的查询"
        )
    else:
        # 非流式 就直接调用 图对象方法
        state: QueryGraphState = invoke_query_graph(
            session_id=session_id,
            original_query=query,
            is_stream=is_stream
        )

        # 这里获取done_list，返回给前端
        done_list = get_done_task_list(session_id)
        # 响应前端
        return QueryNotStreamResponseSchema(
            message=f"完成{query}所有内容检索",
            session_id=session_id,
            answer=state.get("answer"),
            done_list=done_list,
            image_urls=state.get("image_urls", []),
        )


# 接口4: 流式接口
@app.get("/stream/{session_id}")
def stream(session_id:str,request:Request):

    # request.is_disconnected() 通过request对象可以检查前端是否已经断开!
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream"
    )

# 接口5：获取对话历史记录
@app.get("/history/{session_id}")
def get_history(session_id:str,limit:int=10):
    history_list:list[dict] = history_repository.list_recent(session_id=session_id, limit=limit)
    return HistoryListResponseSchema(
        session_id = session_id,
        items = [
            HistoryItemResponseSchema(
                id=str(item.get("_id")),
                session_id=session_id,
                role=item.get("role"),
                text=item.get("text"),
                rewritten_query=item.get("rewritten_query"),
                item_names=item.get("item_names", []),
                image_urls=item.get("image_urls", []),
                ts=item.get("ts")
            )
            for item in history_list]

    )



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)
