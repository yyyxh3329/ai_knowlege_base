from pydantic import BaseModel


class UploadResponseSchema(BaseModel):
    code: int = 200
    message: str | None = None
    task_ids: list[str]

class StatusResponseSchema(BaseModel):
    code: int = 200
    task_id: str
    status: str
    done_list: list[str]
    running_list: list[str]