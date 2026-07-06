import asyncio
import threading
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.shared.runtime.logger import PROJECT_ROOT

app = FastAPI()

# 定义一个根据id查询的函数  get  /user/id ? keyword=xx
@app.get("/user/{id}")
def get_user(id: int):
    return {"id": id}


class UserSchema(BaseModel):
    account: str = None
    password: str = None


# 定义一个用户登录功能  post  user/login {account:root,password:123456}
@app.post("/user/login")
def user_login(user:UserSchema ):
    print(f"接收到的json数据:{user}")
    return user



# file=value(文件) : UploadFile (真文件) 接收文件的类型  = File(文件的约束 规则)  account:str
@app.post("/user/update")
async def user_update(file: UploadFile = File(...)):
    """
    file: UploadFile 对象，包含以下属性：
      - file.filename: 文件名（如 "document.pdf"）
      - file.content_type: MIME 类型（如 "application/pdf"）
      - file.file: 文件内容（异步文件对象）
      - file.read() 读取文件内容

    File(...): 表示该参数为必填项
    """
    # 字节数据
    content = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content)
    }


# 提供一个接口,返回页面!
# get /html/import -> import.html
@app.get("/html/import")
def html_import():

    html_path_obj:Path = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"

    return FileResponse(
        filename=html_path_obj.name,
        content_disposition_type="inline",
        path=str(html_path_obj),
        media_type="text/html"
    )

# 流式返回接口
async def generate_stream():
    # 模拟流式输出（逐字返回）
    words = ["你", "好", "，", "这", "是", "流", "式", "响", "应"]
    for word in words:
        await asyncio.sleep(0.5)
        yield word.encode("utf-8")  # 流式输出需返回字节流

@app.get("/stream")
def stream_response():
    return StreamingResponse(
        generate_stream(),media_type="text/event-stream"
    )

# 后台任务
# 定义一个耗时任务（普通函数即可）
def process_data(task_id: str, data: str):
    """
    模拟耗时操作（如处理文件、调用 AI 模型等）
    这个函数会在后台异步执行，不阻塞 HTTP 响应
    """
    print(f"本次异步任务所在的线程id:{threading.get_ident()}")
    print(f"[{task_id}] 开始处理数据: {data}")
    time.sleep(5)  # 模拟耗时 5 秒
    print(f"[{task_id}] 数据处理完成")
    # 可以在这里更新数据库、写入文件等

@app.get("/start_task")
async def start_task(background_task: BackgroundTasks):
    """
        background_tasks: FastAPI 提供的后台任务管理器
        add_task(): 将任务加入后台队列
        """
    task_id = "task-001"
    print(f"本次接口调用所在的线程id:{threading.get_ident()}")
    # 将耗时任务加入后台队列
    # 注意：这里只是"注册"任务，不会立即执行
    # 执行后端解析流程! 调用了函数
    # add_task(
    #     参数1: 要异步执行的 函数 函数名
    #     参数2 - 参数n: 执行函数的参数
    # )
    background_task.add_task(process_data,task_id=task_id,data="你好，大模型")

    # 立即返回响应，不需要等待 process_data 执行完毕
    return {
        "task_id": task_id,
        "message": "Task started in background"
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)