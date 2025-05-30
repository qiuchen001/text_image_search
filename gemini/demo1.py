import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # 读取图片文件
    image_path = r"E:\playground\ai\datasets\bdd100k\100K\bdd100k_images_bak\bdd100k\images\100k\train\0000f77c-62c2a288.jpg"
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # 读取JSON文件
    json_path = r"E:\playground\ai\datasets\bdd100k\100K\bdd100k_labels\bdd100k\labels\100k\train\0000f77c-62c2a288.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    model = "gemini-2.5-pro-preview-05-06"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=json.dumps(json_data, ensure_ascii=False)),
            ],
        ),

        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text="""你是一名专注于分析和描述车辆前方视角复杂驾驶环境的AI专家。
    阅读并理解已经标注的文字内容，标注的内容主要集中在 frames > objects 以及attributes中的元素，结合提供的图片，生成一系列对图片多维度的简短描述，以列表的形式输出，如：[\"天气是晴朗的\", \"驾驶在城市街道上\", \" 时间是夜晚\", \"当前车道正前方的交通信号灯是红灯\"， \"xxx 更多的描述如有\"]

    注意：
    1. 输出的每一项字数保持在5到20字之间
    2. 你的所有输出必须严格来自于已标注的数据，给你提供图片只是为了让你可以更好的描述已标注的数据"""),
        ],
    )

    response = client.models.generate_content(
        model=model,
        config=generate_content_config,
        contents=contents,
    )

    short_caption_list = json.loads(response.text)
    for item in short_caption_list:
        print(item)


if __name__ == "__main__":
    generate()
