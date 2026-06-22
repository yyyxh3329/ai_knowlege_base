from dotenv import load_dotenv
from langgraph.constants import START
from langgraph.graph import StateGraph, END

from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus
from app.shared.runtime.logger import logger

# 1、定义图的构建对象，并且指定全局state
import_graph_builder = StateGraph(ImportGraphState)

# 2、添加图的节点
import_graph_builder.add_node(node_entry)
import_graph_builder.add_node(node_pdf_to_md)
import_graph_builder.add_node(node_md_img)
import_graph_builder.add_node(node_document_split)
import_graph_builder.add_node(node_item_name_recognition)
import_graph_builder.add_node(node_bge_embedding)
import_graph_builder.add_node(node_import_milvus)

# 3、设置起始节点
import_graph_builder.set_entry_point("node_entry")


# 4、起始节点后的条件边设置
def node_entry_after(state: ImportGraphState):
    """
      判断类型文件  is_md_read_enabled = True  or  is_pdf_read_enabled = True
    :param state:
    :return: 目标节点名称
    """
    if state.get("is_md_read_enabled", False):
        # 是md
        # 日志: 核心点
        # 日志: 交代清楚 有理有据有目标
        logger.info(f"传入的文件地址是{state.get('local_file_path')}，判断传入的文件是md类型，所以跳转到node_md_img")
        return "node_md_img"
    elif state.get("is_pdf_read_enabled", False):
        logger.info(f"传入的文件地址是{state.get('local_file_path')}，判断传入的文件是pdf类型，所以跳转到node_pdf_to_md")
        return "node_pdf_to_md"
    else:
        logger.warning(f"传入的文件地址是{state.get('local_file_path')}，不支持该文件类型处理，只能处理md / pdf格式数据")
        return END


"""
  添加条件边
     参数1: 起始节点 str 节点名
     参数2: 路由函数 state -> 业务逻辑 -> return "节点名称","节点名称","节点名称"
     参数3: path_map dict 显示的配置路由关系! 供静态打印使用
"""
import_graph_builder.add_conditional_edges("node_entry", node_entry_after,{
    "node_md_img":"node_md_img",
    "node_pdf_to_md":"node_pdf_to_md",
    END:END
})

# 5、设置静态边
import_graph_builder.add_edge("node_pdf_to_md", "node_md_img")
import_graph_builder.add_edge("node_md_img","node_document_split")
import_graph_builder.add_edge("node_document_split","node_item_name_recognition")
import_graph_builder.add_edge("node_item_name_recognition","node_bge_embedding")
import_graph_builder.add_edge("node_bge_embedding","node_import_milvus")
import_graph_builder.add_edge("node_import_milvus",END)

# 6、编译图对象
import_app = import_graph_builder.compile()


