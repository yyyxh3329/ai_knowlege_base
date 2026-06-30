from app.infra.config.providers import infra_config
from app.shared.clients import get_milvus_client


class MilvusGateway:
    # 提供两个属性

    # 提供chunks写入的集合名
    def chunk_collection_name(self):
        return infra_config.milvus_config.chunks_collection

    # 提供item_name写入的集合名
    def item_name_collection_name(self):
        return infra_config.milvus_config.item_name_collection

    # 提供一个客户端
    def milvus_client(self):
        return get_milvus_client()


milvus_gateway = MilvusGateway()