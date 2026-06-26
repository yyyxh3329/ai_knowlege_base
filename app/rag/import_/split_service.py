import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger, step_log

@step_log("load_markdown_content")
def load_markdown_content(state: ImportGraphState):
    md_content = state.get("md_content")
    file_title = state.get("file_title")
    md_path = state.get("md_path")

    if not md_content:
        # 如果md_content为空，此时可以从备份地址获取数据
        if md_path and Path(md_path).exists():
            logger.warning(f"md_content内容为空，从备份地址{md_path}读取数据")
            md_content = Path(md_path).read_text()
        if not md_content:
            # 此时从备份地址读取依然为空
            logger.error(f"md_content为空,尝试从md_path读取,依然为空,业务无法继续进行,提前终止!")
            raise ValueError(f"md_content为空,尝试从md_path读取,依然为空,业务无法继续进行,提前终止!")

    if not file_title:
        if md_path and Path(md_path).exists():
            file_title = Path(md_path).stem
        if not file_title:
            file_title = "default"
        state["file_title"] = file_title
        logger.warning(f"file_title为空,启动默认值机制,赋值后:{file_title}")

    # 数据清晰 统一换行符号
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    state["md_content"] = md_content
    return md_content, file_title

@step_log("split_chunks_document")
def split_chunks_document(md_content, file_title) -> list[dict[str, Any]]:
    md_content_lines = md_content.split("\n")

    chunks: list[dict[str, Any]] = []  # [{title,content,file_title}
    current_title: str = None  # 标题
    current_title_lines: list[str] = []  # 标题行
    is_code_block = False
    title_reg = re.compile(r"^\s*#{1,6}\s.+")

    for content_line in md_content_lines:
        # 遇到空行
        line_strip = content_line.strip()
        if not line_strip:
            logger.warning("处理碰到空行!跳过本次处理!")
            continue
        # 这里判断是不是代码块
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            is_code_block = not is_code_block
            current_title_lines.append(content_line)
            continue

        # 如果匹配上了是标题
        if not is_code_block and title_reg.match(line_strip):

            # 表示有标题并且不是第一次来，有内容，所以此时的list里的数量要>1，因为标题也在里面，有内容的化就是>1
            if current_title and len(current_title_lines) > 1:
                chunks.append({
                    "title": current_title,
                    "file_title": file_title,
                    "content": "\n".join(current_title_lines),
                })

            # 没有标题，表示第一次来，并且当前标题的内容数要大于0，等于0 表示这个标题没有内容，进行丢弃
            if not current_title and len(current_title_lines) > 0:
                current_title_lines.append(line_strip)
            else:
                # 1、表示第一次标题来但是没有内容，此时将标题放入list中
                # 2、表示有标题来，但是没有内容，也就是md一开始就是标题
                current_title_lines = [line_strip]

            # 将标题重置
            current_title = line_strip

        else:
            # 没匹配上，也就是 内容或者代码块中的内容，直接添加到对应标题中的list中
            current_title_lines.append(line_strip)

    # 最后一次可能没有被结算
    if current_title and len(current_title_lines) > 1:
        chunks.append({
            "title": current_title,
            "file_title": file_title,
            "content": "\n".join(current_title_lines),
        })

    # 还有一种场景，整个md文档中，就没有标题，
    if len(chunks) == 0 and len(current_title_lines) > 0:
        chunks.append({
            "title": current_title,
            "file_title": "default",
            "content": "\n".join(current_title_lines),
        })

    logger.info(f"完成了语义标题切割，切块数量为:{len(chunks)}")
    return chunks

@step_log("_split_long_chunk")
def _split_long_chunk(chunk):
    """
          主要目标: 过长的进行短切...
          使用技术: langchain递归切割器..
        :param chunk:
        :return:
        """
    """
       chunk
           title # xxxx
           file_tile hk180烫金机
           content  "\n".json(current_title_lines) -> 1. # xxxx\n2. 内容1 ||| 3.内容2 
       注意: 不能只让第一部分有标题 
          1. 先将content标题移除
          2. 定义标准标题 prefix title\n
    """
    # 1、清洗原有的content去掉标签前缀
    content = chunk.get("content")
    title = chunk.get("title")
    file_title = chunk.get("file_title")
    # content  "\n".json(current_title_lines) -> 1. # xxxx\n2. 内容1 ||| 3.内容2
    # 将content中的最前面标题 移除掉
    clear_content = content[len(title) + 1:]
    # 2. 定义要拼接的前缀
    sub_content_prefix = title + "\n"

    # 3. 定义切割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - len(sub_content_prefix), # 扣除标题前缀长度
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";"]
    )

    sub_chunk_list = []

    # 4. 使用递归切割器进行content切割
    for part_index, split_text in enumerate(splitter.split_text(clear_content), start=1):
        # 拼接每个text
        split_text = split_text.strip()
        content_sub_new = sub_content_prefix + split_text
        sub_chunk_list.append({
            "content": content_sub_new,
            "file_title": file_title,
            "part": part_index,
            "parent_title": title,
            "title": f"{title}的第{part_index}部分",
        })
    logger.info(f"进入标题:{title},完成切割后,切成:{len(sub_chunk_list)}块!")
    return sub_chunk_list

@step_log("_merge_short_chunks_same_parent_title")
def _merge_short_chunks_same_parent_title(refine_list):
    """
         同一个标题下,过短(400)进行合并,不超过(1000)
        :param refine_list:
        :return:
        """
    """
      先指向一个基础(pre),作为参照! 
      如果base小于400,尝试将后面的合并入.. 
      合并的前提是: base < 400  同一个parent_title  合并后小于1000 
    """
    merged_chunk_list = []
    # 1、定义一个合并的base chunk 变量
    base_chunk = None  # 要合并入的 不动 判断400  [1]
    # 2.循环处理chunk_list挪动要合并的元素
    for next_chunk in refine_list:
        # 第一次给base_chunk赋值
        if not base_chunk:
            base_chunk = next_chunk
            logger.info(f"短合并第一次进入,设置base_chunk内容!")
            continue

        # 判断是否能被合并
        is_short_chunk = len(base_chunk.get("content")) < CHUNK_MIN
        # 判断是否同一个标题
        is_same_parent_title = base_chunk.get("parent_title") and next_chunk.get("parent_title") == base_chunk.get("parent_title")

        # 满足合并条件： 1、是短chunk，同一个标题
        if is_short_chunk and is_same_parent_title:

            next_content = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 1:]
            # 两个加起来要小于设定的1000，才可以合并
            if (len(next_content) + len(base_chunk.get("content"))) <= CHUNK_MAX_SIZE:
                # 进行合并
                base_chunk["content"] = base_chunk.get("content") + "\n" + next_content
                continue
            else:
                # 不能合并 base 大于400 要不然不是同一个标题
                # base <-- next
                merged_chunk_list.append(base_chunk)
                base_chunk = next_chunk  # 切换指向! next作为基础判断
                continue
        else:
            # 不能合并 base 大于400 要不然不是同一个标题
            # base <-- next
            merged_chunk_list.append(base_chunk)
            base_chunk = next_chunk  # 切换指向! next作为基础判断
            continue

    if base_chunk:
        merged_chunk_list.append(base_chunk)
    logger.info(f"进行短合并,合并之前:{len(refine_list)},合并之后:{len(merged_chunk_list)}")
    return merged_chunk_list




@step_log("refine_chunks")
def refine_chunks(chunks):
    """
    进行精细切割!
      长 -> 600 -> 短切
      短 -> 400 -> 合并 -> 1000
    :param chunks:
    :return:
    """
    # 1. 定义接收最终结果的list
    refine_list = []
    # 2. 进行循环处理查看是否过长
    for chunk in chunks:
        content = chunk.get("content")
        if len(content) > CHUNK_SIZE:
            # 进行长切
            long_chunk_list = _split_long_chunk(chunk)
            refine_list.extend(long_chunk_list)
        else:
            refine_list.append(chunk)

    # 3. 进行短合并处理
    refine_list = _merge_short_chunks_same_parent_title(refine_list)
    # 4. 补全属性..
    for chunk in refine_list:
        if "parent_title" not in chunk:
            chunk['parent_title'] = chunk.get('title', "default_title")
        if "part" not in chunk:
            chunk['part'] = 1
    # 5. 最终返回结果
    logger.info(f"完成chunks的精细处理! 进入切块数量:{len(chunks)},处理后:{len(refine_list)}")
    return refine_list

@step_log("backup_chunks_json")
def backup_chunks_json(refine_chunks_list, md_path:str):
    """
        数据备份
        :param refine_chunks_list:
        :param param:
        :return:
        """
    # 1. 获取目标的地址  文件夹 / 文件名_new.json
    json_path_obj: Path = Path(md_path).with_name(f"{Path(md_path).stem}.json")
    # 2. 目标位置写入字符串
    json_path_obj.write_text(json.dumps(refine_chunks_list, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info(f"已经将切片数据,备份到{json_path_obj}位置!!")


@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """
    # 1.获取参数并且校验
    md_content, file_title = load_markdown_content(state)
    # 2.确保语义切割，根据标题切
    chunks: list[dict[str, Any]] = split_chunks_document(md_content, file_title)
    # 3.进行精细切割
    refine_chunks_list = refine_chunks(chunks)
    # 4. 修改state -> chunks
    state['chunks'] = refine_chunks_list
    # 5. 备份refine_chunks_list  -> [{},{}] -> json字符串 -> 本地磁盘
    backup_chunks_json(refine_chunks_list, state['md_path'])
    return state
