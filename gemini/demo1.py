import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


def process_single_file(json_path, image_path, client):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 读取图片文件
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

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
    print(f"\n处理文件: {os.path.basename(json_path)}")
    for item in short_caption_list:
        print(item)


def generate(max_files=None):
    """
    处理BDD100K数据集中的文件
    :param max_files: 要处理的最大文件数量，None表示处理所有文件
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # 设置路径
    json_dir = r"E:\playground\ai\datasets\bdd100k\100K\bdd100k_labels\bdd100k\labels\100k\train"
    image_dir = r"E:\playground\ai\datasets\bdd100k\100K\bdd100k_images_bak\bdd100k\images\100k\train"

    # 获取所有JSON文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 如果指定了最大文件数，则限制处理数量
    if max_files is not None:
        json_files = json_files[:max_files]
    
    total_files = len(json_files)
    print(f"开始处理，共 {total_files} 个文件")

    # 处理每个JSON文件
    for idx, json_file in enumerate(json_files, 1):
.        json_path = os.path.join(json_dir, json_file)
        # 从JSON文件名中获取图片名称（去掉.json后缀）
        image_name = os.path.splitext(json_file)[0]
        image_path = os.path.join(image_dir, f"{image_name}.jpg")

        print(f"\n[{idx}/{total_files}] 正在处理: {json_file}")

        if os.path.exists(image_path):
            try:
                process_single_file(json_path, image_path, client)
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {str(e)}")
        else:
            print(f"找不到对应的图片文件: {image_path}")


if __name__ == "__main__":
    # 这里可以指定要处理的文件数量，例如处理前5个文件
    generate(max_files=5)
