import os
import jsonlines
from clip_embedding import clip_embedding
from milvus_operator import text_image_vector


def load_timeofday_labels(jsonl_path):
    """加载 imageId 到 timeofday 的映射"""
    id2timeofday = {}
    with jsonlines.open(jsonl_path) as reader:
        for item in reader:
            id2timeofday[item["imageId"]] = item.get("timeofday", None)
    return id2timeofday


def eval_text2image_timeofday_recall_at_k(label_jsonl_path, operator, k=5, query_time='daytime'):
    # 加载所有图片的标签
    id2timeofday = load_timeofday_labels(label_jsonl_path)
    # 用文本生成embedding
    embedding = clip_embedding.embedding_text(query_time)
    embedding = embedding[0].detach().cpu().numpy()
    # 检索
    results = operator.search_data(embedding, limit=k)
    # 统计K个结果中有多少图片的标签是daytime
    hit = 0
    for r in results:
        img_path = r['path']
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        if id2timeofday.get(img_id) is None:
            print(f"{img_id}的标签为None")
            k -= 1
            continue

        if id2timeofday.get(img_id) == query_time:
            hit += 1
        else:
            print(f"{img_id}的标签是{id2timeofday.get(img_id)}, 查询结果是{query_time}")
    recall = hit / k if k > 0 else 0
    print(f"用文本'{query_time}'检索，Recall@{k} = {recall:.4f}，命中{hit}/{k}")


if __name__ == '__main__':
    label_jsonl_path = 'bdd100k_labels_timeofday.jsonl'
    eval_text2image_timeofday_recall_at_k(label_jsonl_path, text_image_vector, k=500, query_time='白天')
