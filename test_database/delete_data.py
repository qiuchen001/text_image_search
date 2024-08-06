from pymilvus import Collection, db, connections

# conn = connections.connect(host="10.66.12.37", port=19530)
# db.using_database("text_image_db")
# coll_name = 'text_image_vector'
#
# collection = Collection(coll_name)
#
# ids = [str(idx) for idx in range(10000)]
# temp_str = ', '.join(ids)
# query_expr = f'm_id in [{temp_str}]'
# result = collection.delete(query_expr)


# 连接到 Milvus 服务器
connections.connect("text_image_db", host='10.66.12.37', port='19530')

# 指定要删除的集合名称
collection_name = "text_image_vector"

# 检查集合是否存在
# if Collection.exists(collection_name):
#     collection = Collection(name=collection_name)
#     # 删除集合
#     collection.drop()
#     print(f"Collection '{collection_name}' has been dropped.")
# else:
#     print(f"Collection '{collection_name}' does not exist.")

collection = Collection(name=collection_name)
# 删除集合
collection.drop()
print(f"Collection '{collection_name}' has been dropped.")