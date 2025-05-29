"""
创建视频集合脚本。

该模块用于在 Milvus 中创建视频集合，定义集合的schema和字段。
包含视频ID、向量embedding、路径、缩略图、摘要和标签等信息。
"""

import os
from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient

# 加载环境变量
load_dotenv()

# 配置 Milvus 连接
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")
DB_NAME = os.getenv("MILVUS_DB_NAME")


def get_milvus_client():
    """
    获取 Milvus 客户端实例
    """
    return MilvusClient(
        uri=uri,
        db_name=DB_NAME,
    )


def _create_collection_schema(milvus_client):
    """
    创建视频集合的schema定义。

    Args:
        milvus_client: Milvus客户端实例

    Returns:
        CollectionSchema: Milvus集合的schema对象
    """
    collection_schema = milvus_client.create_schema(
        auto_id=False,
        enable_dynamic_fields=True,
        description="图文检索集合：存储图片embedding、图片地址、图片摘要、图片标签、图片标题。"
    )

    collection_schema.add_field(
        field_name="m_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=256,
        description="唯一ID"
    )

    collection_schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=3584,
        description="图片embedding"
    )

    collection_schema.add_field(
        field_name="path",
        datatype=DataType.VARCHAR,
        max_length=256,
        description="图片地址"
    )

    collection_schema.add_field(
        field_name="summary_txt",
        datatype=DataType.VARCHAR,
        max_length=3072,
        description="图片摘要",
        nullable=True
    )

    collection_schema.add_field(
        field_name="tags",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=10,
        max_length=256,
        description="图片标签",
        nullable=True
    )

    collection_schema.add_field(
        field_name="title",
        datatype=DataType.VARCHAR,
        max_length=256,
        description="图片标题",
        nullable=True
    )

    collection_schema.add_field(
        field_name="summary_embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=3584,
        description="图片摘要embedding"
    )

    return collection_schema


def _create_collection(milvus_client, collection_schema):
    """
    使用指定的schema创建视频集合。

    Args:
        milvus_client: Milvus客户端实例
        collection_schema (CollectionSchema): 集合的schema定义
    """
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=collection_schema,
    )


def create_collection():
    """
    创建集合的主函数
    """
    milvus_client = get_milvus_client()
    list_collections = milvus_client.list_collections()
    if COLLECTION_NAME in list_collections:
        print("集合已存在")
    else:
        _create_collection(milvus_client, _create_collection_schema(milvus_client))
        print(f"集合 {COLLECTION_NAME} 创建成功")


if __name__ == "__main__":
    create_collection()
