from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection, db, connections


conn = connections.connect(host="10.66.12.37", port=19530)
db.using_database("text_image_db")

m_id = FieldSchema(name="m_id", dtype=DataType.INT64, is_primary=True,)
embeding = FieldSchema(name="embeding", dtype=DataType.FLOAT_VECTOR, dim=768,)
path = FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256,)
schema = CollectionSchema(
  fields=[m_id, embeding, path],
  description="text to image embeding search",
  enable_dynamic_field=True
)

collection_name = "text_image_vector"
collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)