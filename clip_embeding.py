import torch
# import cn_clip.clip as clip
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from PIL import Image
from torchvision import transforms


class ClipEmbedding:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        # self.model, self.processor = clip.load(r"E:\workspace\ai-ground\models\ViT-L-14-336px.pt", device=self.device) # open-ai-clip
        self.model, self.processor = load_from_name(r"E:\playground\ai\models\clip_cn_vit-l-14-336.pt",
                                                    device=self.device, vision_model_name="ViT-L-14-336",
                                                    text_model_name="RoBERTa-wwm-ext-base-chinese",
                                                    input_resolution=336)  # chinese-clip
        self.model.eval()  # chinese-clip

        self.tokenizer = clip.tokenize

        self.transform = transforms.Compose([transforms.Resize((336, 336)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def probs(self, image: Image):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(["a diagram", "a dog", "a cat"]).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(process_image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)

    def match(self, image: Image, desc: str):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer([desc]).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(process_image, text)
            similarity = str(logits_per_image)[9:13]
            res = float(similarity)
            return res

    def embedding_image(self, image: Image):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(process_image)
        return image_features

    def embedding_text(self, text: str):
        text = self.tokenizer([text]).to(self.device)
        text_features = self.model.encode_text(text)
        return text_features

    def embedding(self, image: Image, text: str):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer([text]).to(self.device)

        image_features = self.model.encode_image(process_image)
        text_features = self.model.encode_text(text)
        return image_features, text_features


clip_embedding = ClipEmbedding()

if __name__ == "__main__":
    image_path = 'data/21487e8e0970dd366dafaed6ab25d8d8.jpg'

    pil_image = Image.open(image_path)
    # clip_embedding.probs(pil_image)

    # match = clip_embedding.match(pil_image, "a cat")
    # print(match)

    image_embeddings = clip_embedding.embedding_image(pil_image)
    print(len(image_embeddings[0]))

    # res = image_embeddings[0].detach().numpy().tolist()
    #
    # print(type(res))
    #
    # print(res)

    # embedding = clip_embedding.embedding_text("a cat")
    # print(len(embedding[0]))
