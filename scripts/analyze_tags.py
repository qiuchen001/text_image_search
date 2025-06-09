import json
from collections import Counter
from typing import Dict, List
from pathlib import Path
import json

def analyze_tags(jsonl_file: str) -> Dict:
    """
    分析JSONL文件中的标签分布
    
    Args:
        jsonl_file: JSONL文件路径
        
    Returns:
        包含分析结果的字典
    """
    # 初始化计数器
    object_counter = Counter()
    weather_counter = Counter()
    scene_counter = Counter()
    timeofday_counter = Counter()
    
    # 读取并分析JSONL文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # 统计对象类别
            object_counter.update(data['objects'])
            
            # 统计属性
            weather_counter[data['attributes']['weather']] += 1
            scene_counter[data['attributes']['scene']] += 1
            timeofday_counter[data['attributes']['timeofday']] += 1
    
    # 整理结果
    result = {
        "objects": {
            "total_categories": len(object_counter),
            "distribution": dict(object_counter.most_common())
        },
        "attributes": {
            "weather": {
                "total_categories": len(weather_counter),
                "distribution": dict(weather_counter.most_common())
            },
            "scene": {
                "total_categories": len(scene_counter),
                "distribution": dict(scene_counter.most_common())
            },
            "timeofday": {
                "total_categories": len(timeofday_counter),
                "distribution": dict(timeofday_counter.most_common())
            }
        }
    }
    
    return result

def save_analysis(result: Dict, output_file: str):
    """
    保存分析结果到JSON文件
    
    Args:
        result: 分析结果
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

def main():
    # 设置输入输出路径
    script_dir = Path(__file__).parent
    input_file = script_dir / "extracted_tags.jsonl"
    output_file = script_dir / "tag_analysis.json"
    
    # 分析标签
    result = analyze_tags(str(input_file))
    
    # 保存结果
    save_analysis(result, str(output_file))
    
    # 打印摘要信息
    print("\n=== 标签分析结果摘要 ===")
    print(f"\n对象类别总数: {result['objects']['total_categories']}")
    print("\n前10个最常见的对象类别:")
    for obj, count in list(result['objects']['distribution'].items())[:10]:
        print(f"  {obj}: {count}")
    
    print(f"\n天气类型总数: {result['attributes']['weather']['total_categories']}")
    print("天气分布:")
    for weather, count in result['attributes']['weather']['distribution'].items():
        print(f"  {weather}: {count}")
    
    print(f"\n场景类型总数: {result['attributes']['scene']['total_categories']}")
    print("场景分布:")
    for scene, count in result['attributes']['scene']['distribution'].items():
        print(f"  {scene}: {count}")
    
    print(f"\n时间段类型总数: {result['attributes']['timeofday']['total_categories']}")
    print("时间段分布:")
    for time, count in result['attributes']['timeofday']['distribution'].items():
        print(f"  {time}: {count}")
    
    print(f"\n详细分析结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 