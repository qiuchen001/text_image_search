from pymilvus import connections, db

conn = connections.connect(host="10.66.8.51", port=19530)
# database = db.create_database("sample_db")


db.using_database("sample_db")
res = db.list_database()
print(res)

