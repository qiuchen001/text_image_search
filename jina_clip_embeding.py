from transformers import AutoModel


class JinaClipEmbedding:
    def __init__(self):
        self.model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)

    def embedding_image(self, image):
        image_features = self.model.encode_image(image)
        return image_features

    def embedding_text(self, text: str):
        text_features = self.model.encode_text(text)
        return text_features


clip_embedding = JinaClipEmbedding()

if __name__ == "__main__":
    image_path = 'https://i.pinimg.com/600x315/21/48/7e/21487e8e0970dd366dafaed6ab25d8d8.jpg'

    image_embeddings = clip_embedding.embedding_image([image_path])

    print(len(image_embeddings[0]))
    res = image_embeddings[0].tolist()

    print(type(res))

    print(res)
