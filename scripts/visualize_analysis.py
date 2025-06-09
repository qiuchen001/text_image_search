import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_analysis(json_file: str) -> dict:
    """加载分析结果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_object_distribution(data: dict, output_dir: Path):
    """绘制对象分布图"""
    objects = data['objects']['distribution']
    # 只取前15个对象进行展示
    top_objects = dict(list(objects.items())[:15])
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(top_objects)), list(top_objects.values()))
    plt.xticks(range(len(top_objects)), list(top_objects.keys()), rotation=45, ha='right')
    plt.title('前15个最常见对象的分布')
    plt.xlabel('对象类别')
    plt.ylabel('出现次数')
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'object_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_weather_distribution(data: dict, output_dir: Path):
    """绘制天气分布饼图"""
    weather = data['attributes']['weather']['distribution']
    
    plt.figure(figsize=(10, 10))
    plt.pie(weather.values(), labels=weather.keys(), autopct='%1.1f%%')
    plt.title('天气分布')
    plt.axis('equal')
    plt.savefig(output_dir / 'weather_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_scene_distribution(data: dict, output_dir: Path):
    """绘制场景分布饼图"""
    scene = data['attributes']['scene']['distribution']
    
    plt.figure(figsize=(10, 10))
    plt.pie(scene.values(), labels=scene.keys(), autopct='%1.1f%%')
    plt.title('场景分布')
    plt.axis('equal')
    plt.savefig(output_dir / 'scene_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_timeofday_distribution(data: dict, output_dir: Path):
    """绘制时间段分布饼图"""
    timeofday = data['attributes']['timeofday']['distribution']
    
    plt.figure(figsize=(10, 10))
    plt.pie(timeofday.values(), labels=timeofday.keys(), autopct='%1.1f%%')
    plt.title('时间段分布')
    plt.axis('equal')
    plt.savefig(output_dir / 'timeofday_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_distributions(data: dict, output_dir: Path):
    """绘制所有分布图"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制各种分布图
    plot_object_distribution(data, output_dir)
    plot_weather_distribution(data, output_dir)
    plot_scene_distribution(data, output_dir)
    plot_timeofday_distribution(data, output_dir)
    
    print(f"可视化结果已保存到: {output_dir}")

def main():
    # 设置输入输出路径
    script_dir = Path(__file__).parent
    input_file = script_dir / "tag_analysis.json"
    output_dir = script_dir / "visualization_results"
    
    # 加载分析结果
    data = load_analysis(str(input_file))
    
    # 绘制所有分布图
    plot_all_distributions(data, output_dir)

if __name__ == "__main__":
    main() 