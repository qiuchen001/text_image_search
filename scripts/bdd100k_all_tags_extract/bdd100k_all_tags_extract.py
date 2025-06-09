import json
import os
from typing import Dict, List, Set
from pathlib import Path
from tqdm import tqdm


def extract_tags(json_file_path: str) -> Dict:
    """
    从BDD100K JSON文件中提取标签信息

    Args:
        json_file_path: JSON文件路径

    Returns:
        包含提取标签的字典
    """
    # 初始化结果字典
    result = {
        "file_name": os.path.basename(json_file_path),  # 添加文件名
        "objects": [],
        "attributes": {
            "weather": "",
            "scene": "",
            "timeofday": ""
        }
    }

    # 用于存储唯一的对象类别
    unique_objects: Set[str] = set()

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # 提取对象类别
            if 'frames' in data:
                for frame in data['frames']:
                    if 'objects' in frame:
                        for obj in frame['objects']:
                            if 'category' in obj:
                                unique_objects.add(obj['category'])

            # 提取属性
            if 'attributes' in data:
                for attr in ['weather', 'scene', 'timeofday']:
                    if attr in data['attributes']:
                        result['attributes'][attr] = data['attributes'][attr]

            # 将集合转换为排序后的列表
            result['objects'] = sorted(list(unique_objects))

    except Exception as e:
        print(f"处理文件 {json_file_path} 时出错: {str(e)}")

    return result


def append_to_jsonl(data: Dict, jsonl_file: str):
    """
    将数据追加到JSONL文件中

    Args:
        data: 要追加的数据
        jsonl_file: JSONL文件路径
    """
    try:
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"写入JSONL文件时出错: {str(e)}")


def process_directory(input_dir: str, output_file: str):
    """
    处理目录下的所有JSON文件

    Args:
        input_dir: 输入目录路径
        output_file: 输出JSONL文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 获取所有JSON文件
    json_files = list(Path(input_dir).glob('*.json'))
    total_files = len(json_files)

    print(f"找到 {total_files} 个JSON文件")

    # 使用tqdm显示进度
    for json_file in tqdm(json_files, desc="处理文件"):
        result = extract_tags(str(json_file))
        append_to_jsonl(result, output_file)


def main():
    # 设置输入和输出路径
    input_dir = r"E:\playground\ai\datasets\bdd100k\100K\bdd100k_labels\bdd100k\labels\100k\train"
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted_tags.jsonl")

    # 处理目录
    process_directory(input_dir, output_file)

    print(f"\n处理完成！结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
