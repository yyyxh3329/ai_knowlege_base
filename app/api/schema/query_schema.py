from pydantic import BaseModel, Field


# 检查健康的响应json
class HealthResponseSchema(BaseModel):
    code: int = 200
    message: str = None


# 查询接口的请求参数json
class QueryRequestSchema(BaseModel):
    session_id: str
    query: str
    is_stream : bool


# 流式查询的立即响应
class QueryStreamResponseSchema(BaseModel):
    session_id: str
    message: str

# 非流式查询的响应
class QueryNotStreamResponseSchema(BaseModel):
    session_id: str
    message: str
    answer:str
    done_list: list[str]
    image_urls: list[str]

class HistoryItemResponseSchema(BaseModel):
    id:str
    session_id: str
    role:str
    text:str
    rewritten_query:str
    item_names:list[str] = Field(description="关联的item_name",default_factory=list)
    image_urls:list[str] = Field(description="关联的图片地址",default_factory=list)

# 查询历史对话记录的响应
class HistoryListResponseSchema(BaseModel):
    session_id: str
    items:list[HistoryItemResponseSchema] = Field(description="查询的数据记录列表",default_factory=list)