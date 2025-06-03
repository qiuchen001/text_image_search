import os
import jsonlines
from PIL import Image
from clip_embedding import clip_embedding
from milvus_operator import text_image_vector, MilvusOperator


def embed_images_from_jsonl(jsonl_path, image_dir, operator: MilvusOperator, batch_size=50):
    idxs, embeddings, paths = [], [], []
    total_count = 0
    with jsonlines.open(jsonl_path) as reader:
        for item in reader:
            image_id = item["imageId"]
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                print(f"图片不存在: {image_path}")
                continue
            try:
                image = Image.open(image_path).convert('RGB')
                embedding = clip_embedding.embedding_image(image)
                idxs.append(total_count)
                embeddings.append(embedding[0].detach().cpu().numpy().tolist())
                paths.append(image_path)
                total_count += 1
                if total_count % batch_size == 0:
                    data = [idxs, embeddings, paths]
                    operator.insert_data(data)
                    print(f'success insert {operator.coll_name} items:{len(idxs)}')
                    idxs, embeddings, paths = [], [], []
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {str(e)}")
    # 插入剩余
    if len(idxs):
        data = [idxs, embeddings, paths]
        operator.insert_data(data)
        print(f'success insert {operator.coll_name} items:{len(idxs)}')
    print(f'finish update {operator.coll_name} items: {total_count}')


if __name__ == '__main__':
    jsonl_path = r'bdd100k_captions.jsonl'
    image_dir = r'E:\playground\ai\datasets\bdd100k\100K\bdd100k_images\bdd100k\images\100k\train'
    embed_images_from_jsonl(jsonl_path, image_dir, text_image_vector)
