import shutil
import time
from pathlib import Path

import requests

from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log
from app.rag.import_.config import PDF_PARSE_SERVICE_LOCAL_DIR, MINERU_POLL_INTERVAL_SECONDS, \
    MINERU_DOWNLOAD_TIMEOUT_SECONDS


# 1. 参数获取和校验 (state -> str)-> (pdf_path:Path,local_dir:Path)  validate_pdf_paths
@step_log("validate_pdf_paths")
def validate_pdf_paths(state: ImportGraphState) -> tuple[Path, Path]:
    # 1.1 state获取 pdf_path 和 local_dir : str
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")
    # 1.2 进行pdf_path非空校验
    #     空 -> 打印日志error -> 异常 ValueError
    if not pdf_path:
        logger.error(f"pdf_path参数为空,业务无法继续进行,提前终止!")
        raise ValueError(f"pdf_path参数为空,业务无法继续进行,提前终止!")
    # 1.3 进行local_path非空校验
    #     空 -> 打印日志warning -> 给与默认值  项目根路径 / output
    if not local_dir:
        logger.warning(f"local_dir参数为空,我们给与默认值,默认为: 项目根地址/output 文件夹!")
        """
            Path  name stem suffix  read... writer... 
                  拼接地址 Path  / Path or "xxx"
                  local_dir_obj = PROJECT_ROOT / "output" / "xxx"  -> 推荐
        """
        local_dir: Path = PROJECT_ROOT / PDF_PARSE_SERVICE_LOCAL_DIR
        state["local_dir"] = str(local_dir)
    # 1.4 将pdf_path local_dir 转成Path
    pdf_path_obj: Path = Path(pdf_path)
    local_dir_obj: Path = Path(local_dir)

    # 1.5 pdf_path_obj:Path 判断是否存在
    #     不存在,打印日志error -> 异常 FileNotFoundError
    if not pdf_path_obj.exists():
        logger.error(f"存在pdf_path地址:{str(pdf_path_obj)},但是地址没有对应文件,业务无法继续进行,提前终止!")
        raise FileNotFoundError(f"存在pdf_path地址:{str(pdf_path_obj)},但是地址没有对应文件,业务无法继续进行,提前终止!")

    # 1.6 local_dir_obj:Path 是不是目录(is_dir())
    #     不是(不存在,或者不是目录)
    #     打印日志warning -> 创建 local_dri_obj对应的文件夹
    if not local_dir_obj.is_dir():
        logger.warning(f"存在local_dir地址:{str(local_dir_obj)},但是没有对应的文件夹,我们创建对应的文件夹,业务继续!")
        # parents 如果有多层文件夹也会一定创建  就是你们在学linux命令的时候 mkdir -P x/x/x/x/x
        # exist_ok 如果存在,也不会报错,也不重复创建!
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    # 1.7 返回结果 return pdf_path_obj , local_dir_obj
    return pdf_path_obj, local_dir_obj

@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj) -> str:
    # 2.1 向minerU服务器发送请求申请上传地址 (batch_id / url)

    url = "https://mineru.net/api/v4/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {infra_config.mineru_config.api_key}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": "vlm"
    }

    response = requests.post(url, headers=header, json=data)
    # 进行校验  所有的网络请求,必须两步骤判定 1. 状态码 200 http的网络状态  2. 业务状态必须成功
    if response.status_code != 200:  # 1. 状态码 200 http的网络状态
        logger.error(f"向minerU服务器申请上传文件解析,但是http状态码为:{response.status_code},状态错误,无法继续业务!")
        raise RuntimeError(
            "向minerU服务器申请上传文件解析,但是http状态码为:{response.status_code},状态错误,无法继续业务!")
    response_dict = response.json()  # 获取响应体的字符串，并转换为dict
    if response_dict.get("code", -1) != 0:  # 2. 业务状态必须成功
        logger.error(
            f"向minerU服务器申请上传文件解析,网络状态正常,服务业务状态异常,code = {response_dict.get('code', -1)}"
            f"错误原因:{response_dict.get('msg')},无法继续业务!")
        raise RuntimeError(
            f"向minerU服务器申请上传文件解析,网络状态正常,服务业务状态异常,code = {response_dict.get('code', -1)}"
            f"错误原因:{response_dict.get('msg')},无法继续业务!")
    # 此时网络没有问题，业务也没有问题
    batch_id = response_dict.get("data", {}).get("batch_id")
    file_upload_urls = response_dict.get('data', {}).get('file_urls', [])
    file_upload_url = None
    if len(file_upload_urls) > 0:
        file_upload_url = file_upload_urls[0]

    if not batch_id:
        logger.error(f"申请minerU解析文件，返回的batch_id为空，业务无法继续执行！业务中断！")
        raise ValueError(f"申请minerU解析文件，返回的batch_id为空，业务无法继续执行！业务中断！")

    logger.info(f"完成文件上传申请，batch_id:{batch_id},预签名地址file_upload_url:{file_upload_url}")

    # 2.2 向指定的url地址发起网络请求并且上传pdf文件
    with requests.Session() as session:
        # session (1.复用请求请求对象 2. 属性设置了以后,可以不信任当前系统的环境,保证请求的整洁性) 和 requests 都可以发起请求
        session.trust_env = False
        upload_response = session.put(url=file_upload_url, data=pdf_path_obj.read_bytes())
        # 此时只用判断http响应状态码 200，不用判断业务状态码，因为不是一个接口，只是一个文件服务器，只有网络状态码，没有业务
        if response.status_code != 200:
            logger.error(
                f"向minerU文件服务器地址{pdf_path_obj}上传文件，返回的状态码为{upload_response.status_code}，业务失败，提前终止")
            raise RuntimeError(
                f"向minerU文件服务器地址{pdf_path_obj}上传文件，返回的状态码为{upload_response.status_code}，业务失败，提前终止")

    # 2.3 轮询向minerU获取batch_id解析状态 zip_url
    # 获取minerU解析结果
    # 方案1: 回调 (minerU -> 我们的服务器 fastapi)  申请地址的时候 请求体中 callback = 我们的地址
    # 方案2: 轮询 (我们 -> 3s -> minerU -> batch_id -> 解析结果) [我们]
    # 准备数据格式
    result_url = f"{infra_config.mineru_config.base_url}/extract-results/batch/{batch_id}"
    start_time = time.time()
    # 声明一个循环
    while True:
        # 1、先判断时间 是否超时 600
        if time.time() - start_time >= 600:
            logger.error(f"轮询获取{batch_id}对应的解析结果超时！耗时为：{time.time() - start_time}")
            raise TimeoutError(f"轮询获取{batch_id}对应的解析结果超时！耗时为：{time.time() - start_time}")
        # 2、没有超时就向接口发起请求获取解析结果
        try:
            poll_result = requests.get(url=result_url, headers=header)
        except requests.exceptions.RequestException as e:
            logger.warning(f"申请结果出现波动{str(e)},稍后再试")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
        # 3、网络状态判定
        if poll_result.status_code != 200:
            # 分情况如果是500 minerU服务端出现问题，我们要给机会重试，如果是400及其他状态码就是我们客户端出现问题就不给机会
            if 500 <= poll_result.status_code < 600:
                logger.warning(f"申请结果出现网络状态错误：{poll_result.status_code},稍后再试，等待服务器修复！")
                time.sleep(MINERU_POLL_INTERVAL_SECONDS)
                continue
            else:
                logger.error(
                    f"获取{batch_id}对应的解析结果，服务器访问报错,http状态码：{poll_result.status_code},错误无法修复，业务无法进行，提前终止")
                raise RuntimeError(
                    f"获取{batch_id}对应的解析结果，服务器访问报错,http状态码：{poll_result.status_code},错误无法修复，业务无法进行，提前终止")
        # 4、业务状态判定
        poll_result_dict = poll_result.json()
        if poll_result_dict.get("code", -1) != 0:
            # 业务失败 不给机会
            logger.error(
                f"获取{batch_id}对应的解析结果，业务状态报错！业务状态码：{poll_result_dict.get('code', -1)}，错误信息:{poll_result_dict.get('msg')},业务失败,提前终止!")
            raise RuntimeError(
                f"获取{batch_id}对应的解析结果，业务状态报错！业务状态码：{poll_result_dict.get('code', -1)}，错误信息:{poll_result_dict.get('msg')},业务失败,提前终止!")
        # 5、获取解析结果和状态判定
        extract_result_list = poll_result_dict.get("data", {}).get("extract_result", [])
        if len(extract_result_list) == 0:
            # 错误是可以给机会! 这次不行了
            logger.warning(f"解析结果extract_result_list为空,跳过本次!稍后再试")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
        extract_result = extract_result_list[0]
        state = extract_result.get('state')

        if state == "done":
            full_zip_url = extract_result.get("full_zip_url")
            if not full_zip_url:
                # 不给机会
                logger.error(
                    f"获取:{batch_id}对应的解析结果,任务已经完成,但是full_zip_url没有地址!业务失败,提前终止!")
                raise ValueError(
                    f"获取:{batch_id}对应的解析结果,任务已经完成,但是full_zip_url没有地址!业务失败,提前终止!")
            # 2.4 返回zip_url ...
            return full_zip_url
        elif state == "failed":
            # 解析完毕,失败了
            # 不给机会
            logger.error(
                f"获取:{batch_id}对应的解析结果,任务解析失败!业务失败,提前终止!")
            raise ValueError(
                f"获取:{batch_id}对应的解析结果,任务解析失败!业务失败,提前终止!")
        else:
            # 正在解析中...
            logger.warning("本次解析,没有获得结果,继续下一次!!!")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue

@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_url: str, local_path_obj: Path, file_name: str):
    """
           进行地址下载和解压,以及重命名! 最终返回md_path_obj
        :param zip_url:
        :param local_dir_obj:
        :param file_name:
        :return:
        """
    # 1.下载数据[重试三次]
    response = requests.get(url=zip_url, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    # 这里是是从文件服务器下载，只有http状态码，没有业务状态码
    if response.status_code != 200:
        logger.error(f"向指定地址：{zip_url}地址下载zip文件报错，状态码为{response.status_code}，业务无法继续进行")
        raise RuntimeError(f"向指定地址：{zip_url}地址下载zip文件报错，状态码为{response.status_code}，业务无法继续进行")
    # 2.准备zip文件对象
    zip_file_obj: Path = local_path_obj / f"{file_name}.zip"
    # 3.向文件对象写入数据
    """
          response 
              .status_code 
              .json()  -> 服务器返回的json字符串 -> dict
              .text    -> 服务器返回的json字符串 -> str -> json.loads...
              .content -> 服务器返回的字节数据   
        """
    zip_file_obj.write_bytes(response.content)
    # 4.解压前先 查看是否有对应的文件夹
    zip_extract_dir: Path = local_path_obj / file_name
    if zip_extract_dir.is_dir():
        # 表示之前解压过，要将之前的解压的数据全部删除，递归清空所有数据
        shutil.rmtree(zip_extract_dir)
    # 创建文件夹
    zip_extract_dir.mkdir(parents=True, exist_ok=True)
    # 向文件夹解压数据
    """
       参数1: 要解压的压缩包
       参数2: 解压到的文件夹
      unpack_archive 解压  使用简单,支持所有格式的压缩包
    """
    shutil.unpack_archive(zip_file_obj, zip_extract_dir)
    # 5.对.md文件进行重命名
    # 取出文件夹下所有的md文件
    md_obj_list: list[Path] = list(zip_extract_dir.rglob("*.md"))
    if len(md_obj_list) == 0:
        logger.error(f"向指定地址：{zip_url}下载zip文件，解压后发现没有md文件，业务无法继续进行")
        raise ValueError(f"向指定地址：{zip_url}下载zip文件，解压后发现没有md文件，业务无法继续进行")
    # 情况1: 就是文件名 -> 直接return ..
    for current_md_obj in md_obj_list:
        if current_md_obj.stem == file_name:
            return current_md_obj
    md_obj_path: Path = None
    # 情况2: full -> 记录
    for current_md_obj in md_obj_list:
        if current_md_obj.stem == "full":
            md_obj_path = current_md_obj
            break
    # 情况3: xxxx -> 记录
    if not md_obj_path:
        md_obj_path = md_obj_list[0]
    # rename -> 真的会修改磁盘!!
    logger.info(f"触发了md文件的重命名机制,原名称:{md_obj_path.stem},目标名称:{file_name}")
    md_obj_path.rename(md_obj_path.with_name(f"{file_name}.md"))
    return md_obj_path

@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
      进行pdf转成md业务!
      最后修改state md_path属性
    """
    # 1.获取并且校验参数 validate_pdf_paths(state) -> tuple(pdf_path_obj: Path, local_dir_obj: Path)
    pdf_path_obj, local_path_obj = validate_pdf_paths(state)
    # 2.minerU解析pdf文件并返回zip的下载地址
    zip_url: str = upload_pdf_and_poll(pdf_path_obj)
    # 3.根据zip_url下载并解压和重命名md文件
    md_obj_path = download_and_extract_markdown(zip_url, local_path_obj, pdf_path_obj.stem)
    state['md_path'] = str(md_obj_path)
    return state
