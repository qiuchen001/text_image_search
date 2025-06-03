from clip_embedding import clip_embedding
# from jina_clip_embedding import clip_embedding
from milvus_operator import text_image_vector, MilvusOperator
from PIL import Image
import os


def update_image_vector(data_path, operator: MilvusOperator):
    idxs, embeddings, paths = [], [], []

    total_count = 0
    for dir_name in os.listdir(data_path):
        if dir_name != 'train':
            continue
        sub_dir = os.path.join(data_path, dir_name)
        for file in os.listdir(sub_dir):

            image = Image.open(os.path.join(sub_dir, file)).convert('RGB')
            # embedding = clip_embedding.embedding_image([image]) # jina-clip
            embedding = clip_embedding.embedding_image(image)

            idxs.append(total_count)
            # embeddings.append(embedding[0].tolist()) # jina-clip
            embeddings.append(embedding[0].detach().cpu().numpy().tolist())

            paths.append(os.path.join(sub_dir, file))
            total_count += 1

            if total_count % 50 == 0:
                data = [idxs, embeddings, paths]
                operator.insert_data(data)

                print(f'success insert {operator.coll_name} items:{len(idxs)}')
                idxs, embeddings, paths = [], [], []

        if len(idxs):
            data = [idxs, embeddings, paths]
            operator.insert_data(data)
            print(f'success insert {operator.coll_name} items:{len(idxs)}')

    print(f'finish update {operator.coll_name} items: {total_count}')


if __name__ == '__main__':
    data_dir = r'E:\playground\ai\datasets\bdd100k\100K\bdd100k_images\bdd100k\images\100k'
    update_image_vector(data_dir, text_image_vector)
