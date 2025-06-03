import os
import json
import jsonlines

# 设置BDD100K标签目录和输出文件
json_dir = r"E:\playground\ai\datasets\bdd100k\100K\bdd100k_labels\bdd100k\labels\100k\train"
output_file = "bdd100k_labels_timeofday.jsonl"
captions_jsonl = "bdd100k_captions.jsonl"


def extract_timeofday_from_json(json_path):
    """
    从单个JSON文件中提取图片ID和attributes > timeofday字段
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 假设图片ID为文件名（不含扩展名）
        image_id = os.path.splitext(os.path.basename(json_path))[0]
        attributes = data.get('attributes', {})
        timeofday = attributes.get('timeofday', None)
        return {
            "imageId": image_id,
            "timeofday": timeofday
        }
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        return None


def main():
    # 读取 captions jsonl 中的 imageId
    image_ids = set()
    with jsonlines.open(captions_jsonl) as reader:
        for item in reader:
            image_ids.add(item["imageId"])
    print(f"共需处理 {len(image_ids)} 个 imageId")
    with jsonlines.open(output_file, mode='w') as writer:
        for image_id in image_ids:
            json_path = os.path.join(json_dir, f"{image_id}.json")
            if not os.path.exists(json_path):
                print(f"标签文件不存在: {json_path}")
                continue
            result = extract_timeofday_from_json(json_path)
            if result:
                writer.write(result)
    print(f"已保存到: {output_file}")


if __name__ == "__main__":
    main()
