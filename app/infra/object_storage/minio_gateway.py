# 引入现有的功能,进行汇总!
# 因为项目中有minio的util和config,此时引入一个infra公共类来进行汇总，后面使用只用导入这一个类就好
from minio import Minio

from app.infra.config.providers import infra_config
from app.shared.clients import get_minio_client


class MinioGateway:

    # 提供获取桶名称的函数
    @property
    def bucket_name(self) -> str:
        return infra_config.minio_config.bucket_name

    # 提供获取图片名称的函数
    @property
    def minio_img_dir(self) -> str:
        return infra_config.minio_config.minio_img_dir

    # 提供获取minio客户端的函数
    @property
    def minio_client(self) -> Minio:
        return get_minio_client()

    # 封装一个拼接访问地址的函数
    def build_image_url(self, stem: str, image_name: str) -> str:
        """
          object_name = upload-images / 文件名 / 图片名称.png
        :param image_name:
        :return:
        """
        image_url = "https://" if infra_config.minio_config.minio_secure else "http://" + (f"{infra_config.minio_config.endpoint}"
            f"/{infra_config.minio_config.bucket_name}{infra_config.minio_config.minio_img_dir}/{stem}/{image_name}")

        return image_url

minio_gateway = MinioGateway()

