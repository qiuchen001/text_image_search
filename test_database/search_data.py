from pymilvus import Collection, db, connections
import numpy as np

conn = connections.connect(host="10.66.12.37", port=19530)
db.using_database("text_image_db")
coll_name = 'text_image_vector'

search_params = {
    "metric_type": 'IP',
    "offset": 0,
    "ignore_growing": False,
    "params": {"nprobe": 16}
}

collection = Collection(coll_name)
collection.load()

results = collection.search(
    data=[np.random.normal(0, 0.1, 768).tolist()],
    anns_field="embeding",
    param=search_params,
    limit=16,
    expr=None,
    # output_fields=['m_id', 'embeding', 'desc', 'count'],
    output_fields=['m_id', 'desc', 'count'],
    consistency_level="Strong"
)
collection.release()
print(results[0].ids)
print(results[0].distances)
hit = results[0][0]
print(hit.entity.get('desc'))
print(results)
