from langgraph.graph import StateGraph

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState

# 1.创建一个编译对象
query_graph_builder = StateGraph(QueryGraphState)

# 2.添加节点
query_graph_builder.add_node(node_item_name_confirm)
query_graph_builder.add_node(node_search_embedding)
query_graph_builder.add_node(node_search_embedding_hyde)
query_graph_builder.add_node(node_web_search_mcp)
query_graph_builder.add_node(node_rrf)
query_graph_builder.add_node(node_rerank())
query_graph_builder.add_node(node_answer_output())


