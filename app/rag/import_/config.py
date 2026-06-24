MINERU_MODEL_VERSION = "vlm"
# MinerU 任务轮询最大超时时间（单位：秒），超过则判定任务失败
# 600 -> 一个pdf 约等于 1秒
MINERU_POLL_TIMEOUT_SECONDS = 600
# MinerU 任务轮询间隔时间（单位：秒），每隔多久查询一次任务状态
MINERU_POLL_INTERVAL_SECONDS = 3
# MinerU 文件下载超时时间（单位：秒），下载文件超过此时长则中断
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 30

# 定义local_dir对应输出的常来那个
PDF_PARSE_SERVICE_LOCAL_DIR = "output"