from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
collection_name = os.getenv("MILVUS_COLLECTION_NAME")
DB_NAME = os.getenv("MILVUS_DB_NAME")

EMBEDDING_INDEX = "embedding_index"
SUMMARY_EMBEDDING_INDEX = "summary_embedding_index"


def get_milvus_client():
    """
    获取 Milvus 客户端实例
    """
    return MilvusClient(
        uri=uri,
        db_name=DB_NAME,
    )


def _create_index(milvus_client):
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="FLAT",
        index_name=EMBEDDING_INDEX,
        params={"nlist": 1024}
    )

    index_params.add_index(
        field_name="summary_embedding",
        metric_type="COSINE",
        index_type="FLAT",
        index_name=SUMMARY_EMBEDDING_INDEX,
        params={"nlist": 1024}
    )

    milvus_client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    print(f"索引 {EMBEDDING_INDEX}、{SUMMARY_EMBEDDING_INDEX} 创建成功")


def create_index():
    milvus_client = get_milvus_client()

    list_indexes = milvus_client.list_indexes(
        collection_name=collection_name
    )

    if EMBEDDING_INDEX in list_indexes and SUMMARY_EMBEDDING_INDEX in list_indexes:
        print(f"索引 {EMBEDDING_INDEX}、{SUMMARY_EMBEDDING_INDEX} 已存在")
    else:
        _create_index(milvus_client)
        # 创建后立即加载集合
        milvus_client.load_collection(
            collection_name=collection_name
        )
        print(f"集合 {collection_name} 加载成功")


if __name__ == "__main__":
    create_index()
