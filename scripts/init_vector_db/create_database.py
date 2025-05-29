import os
from pymilvus import MilvusClient
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

DB_NAME = os.getenv("MILVUS_DB_NAME")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

client = MilvusClient(
    uri=MILVUS_URI,
    token="root:Milvus"
)


def create_databases():
    list_databases = client.list_databases()

    if DB_NAME in list_databases:
        print("数据库已存在")
    else:
        client.create_database(
            db_name=DB_NAME
        )
        print("数据库创建成功")


if __name__ == '__main__':
    create_databases()
