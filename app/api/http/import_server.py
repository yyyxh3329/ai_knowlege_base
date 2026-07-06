import shutil
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi import FastAPI, UploadFile, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.import_schema import UploadResponseSchema, StatusResponseSchema
from app.infra.config.providers import infra_config
from fastapi.responses import FileResponse

from app.process.import_.agent.main_graph import import_app
from app.process.import_.agent.state import create_default_state, ImportGraphState
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.shared.utils.task_utils import add_running_task, add_done_task, update_task_status, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, get_done_task_list, get_running_task_list, get_task_status

app = FastAPI(
    title=infra_config.settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.2.0",
)


# CORS跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(infra_config.settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1.接口1 返回html文件
@app.get("/import/html")
def import_html():

    html_import_path:Path = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"

    return FileResponse(
        path=str(html_import_path),
        media_type=guess_type(html_import_path)[0],
    )

def invoke_import_graph(task_id:str,local_file_path:str,local_dir:str):
    # 执行必须要state -> 必须要三个参数  task_id / local_file_path  / local_dir
    try:
        # processing
        update_task_status(task_id, status_name=TASK_STATUS_PROCESSING)
        state: ImportGraphState = create_default_state(
            task_id=task_id, local_file_path=local_file_path, local_dir=local_dir
        )
        import_app.invoke(state)
        # completed
        update_task_status(task_id, status_name=TASK_STATUS_COMPLETED)
    except Exception as e:
        # failed
        update_task_status(task_id, status_name=TASK_STATUS_FAILED)
        logger.exception(f"导入模块执行发生异常!异常信息:{str(e)}")

# 2.接口二
@app.post("/upload")
def uploads(files:list[UploadFile],backgroundtasks:BackgroundTasks):
    """
    本质: 异步调用图对象,解析本次上传的文件
         backgroundtasks.add_task(函数,参数)
    1. 先定一个一个执行graph的函数 (task_id,local_file_path,local_dir)
    2. 凑齐参数 task_id,local_file_path,local_dir
    3. 异步调用 执行graph的函数 并传递参数
    4. 返回前端数据
    :param backgroundtasks:
    :param files:
    :return:
    """
    # 1. 凑齐参数 task_id,local_file_path,local_dir
    # task_id [每个文件的任务标识,解析的时候,使用task_id作为key存储当前文件的解析状态! 前端也会使用task_id查询对应的状态]
    # 生成task_id(唯一不重复)  返回task_id即可
    # uuid -> 时区 / 时间戳 / ip地址 / mac地址
    task_id = str(uuid.uuid4())
    # local_dir ->  项目根路径 / output / 20260626 / task_id
    time_now_str = datetime.now().strftime("%Y%m%d")
    local_dir_obj = PROJECT_ROOT / "output" / time_now_str / task_id
    # 创建这个目录
    local_dir_obj.mkdir(parents=True, exist_ok=True)
    # 接口接受到的文件在内存中，要将文件保存到local_dir_path_obj目录中
    upload_file = files[0]
    local_file_path_obj:Path = local_dir_obj / upload_file.filename

    # 2.上传文件存储到地址存储文件 [后续就可以读取和解析]
    add_running_task(task_id, "upload_file")
    # 优化: 一次读取全部容易导致OOM
    # local_file_path_obj.write_bytes(upload_file.read())
    # 思路: 每次部分读取
    with open(local_file_path_obj, "wb") as file_buffer:
        # copyfileobj 好处：
        # 参数1: 上传文件(数据)
        # 参数2: 要写入的文件引用
        # 循环读取 !  window默认是 1mb 1024 * 1024  非window 64kb
        # 指定每次 10 MB = length = 1 * 1025 kb * 1024 mb * 10
        shutil.copyfileobj(upload_file.file, file_buffer)
    add_done_task(task_id, "upload_file")

    # 3.异步执行
    backgroundtasks.add_task(
        invoke_import_graph,
        task_id=task_id,
        local_file_path=str(local_file_path_obj),
        local_dir=str(local_dir_obj)
    )

    return UploadResponseSchema(
        code=200,
        message=f"{upload_file.filename}文件上传成功!",
        task_ids=[task_id]
    )

# 3.接口3  获取请求状态
@app.get("/status/{task_id}")
def task_status(task_id: str):
    # 每次解析任务内部的节点的调用状态
    # task_utils 1. 定义了存储数据的字典 2.定义了存储数据方法 3. 定义获取数据方法
    done_list = get_done_task_list(task_id)
    running_list = get_running_task_list(task_id)
    # status 这个任务的总状态
    # update_task_status(task_id,status) -> 执行图对象
    # get_task_status
    status = get_task_status(task_id)

    return StatusResponseSchema(
        code=200,
        task_id=task_id,
        status=status,
        done_list=done_list,
        running_list=running_list
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=infra_config.settings.app_host, port=infra_config.settings.import_app_port)