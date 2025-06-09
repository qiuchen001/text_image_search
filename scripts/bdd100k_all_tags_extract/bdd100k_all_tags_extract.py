import json
import os
from typing import Dict, List, Set


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
        print(f"处理文件时出错: {str(e)}")

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


def main():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建输入文件路径
    input_file = os.path.join(script_dir, "00b04b30-2e874876.json")

    # 构建输出JSONL文件路径
    output_file = os.path.join(script_dir, "extracted_tags.jsonl")

    # 提取标签
    result = extract_tags(input_file)

    # 将结果追加到JSONL文件
    append_to_jsonl(result, output_file)

    # 打印结果
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print(f"\n结果已追加到文件: {output_file}")


if __name__ == "__main__":
    main()
