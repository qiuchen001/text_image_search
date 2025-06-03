import os
import json
import jsonlines

# 设置BDD100K标签目录和输出文件
json_dir = r"E:\playground\ai\datasets\bdd100k\100K\bdd100k_labels\bdd100k\labels\100k\train"
output_file = "bdd100k_labels.jsonl"


def extract_labels_from_json(json_path):
    """
    从单个JSON文件中提取图片ID和标签内容
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 假设图片ID为文件名（不含扩展名）
        image_id = os.path.splitext(os.path.basename(json_path))[0]
        # 你可以根据需要自定义提取内容
        # 这里以提取frames、objects、attributes为例
        frames = data.get('frames', [])
        attributes = data.get('attributes', {})
        # 只提取第一个frame的objects（如有）
        objects = []
        if frames and isinstance(frames, list):
            objects = frames[0].get('objects', [])
        return {
            "imageId": image_id,
            "objects": objects,
            "attributes": attributes
        }
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        return None


def main():
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    with jsonlines.open(output_file, mode='w') as writer:
        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            result = extract_labels_from_json(json_path)
            if result:
                writer.write(result)
    print(f"已保存到: {output_file}")


if __name__ == "__main__":
    main()
