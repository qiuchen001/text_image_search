from clip_embeding import clip_embeding
from milvus_operator import restnet_image, MilvusOperator
from PIL import Image
import os


def update_image_vector(data_path, operator: MilvusOperator):
    idxs, embedings, paths = [], [], []

    total_count = 0
    for dir_name in os.listdir(data_path):
        sub_dir = os.path.join(data_path, dir_name)
        for file in os.listdir(sub_dir):

            image = Image.open(os.path.join(sub_dir, file)).convert('RGB')
            embeding = clip_embeding.embeding_image(image)

            idxs.append(total_count)
            embedings.append(embeding[0].detach().numpy().tolist())
            paths.append(os.path.join(sub_dir, file))
            total_count += 1

            if total_count % 50 == 0:
                data = [idxs, embedings, paths]
                operator.insert_data(data)

                print(f'success insert {operator.coll_name} items:{len(idxs)}')
                idxs, embedings, paths = [], [], []

        if len(idxs):
            data = [idxs, embedings, paths]
            operator.insert_data(data)
            print(f'success insert {operator.coll_name} items:{len(idxs)}')

    print(f'finish update {operator.coll_name} items: {total_count}')


if __name__ == '__main__':
    data_dir = 'D:/dataset/fruit'
    update_image_vector(data_dir, restnet_image)

