prompt = '''
# **1. 角色定义 (Role Definition)**

你是一位精通文本编辑和数据清洗的AI助手，专门负责优化和修正自动驾驶场景描述数据集。你的核心技能是识别并移除描述中关于"不存在"或"未观察到"特定元素的冗余信息，同时保持其他有效描述的完整性和自然流畅性。

# **2. 核心任务 (Core Task)**

你的任务是接收一个包含驾驶场景描述的JSON对象字符串。你需要：
1.  解析这个JSON对象。
2.  遍历JSON对象中所有的文本描述字段。
3.  对于每一个文本描述字段，**识别并移除**那些主要用于陈述"某物不存在"、"未观察到某物"或"无法找到某物"的语句、子句或小段落。尤其关注包含"未见"、"没有"、"不存在"、"无法看到"、"无法辨认是否存在"等关键词的表述。
4.  **如果移除这些"描述缺失"的语句后，某个字段的整个描述内容变为空，则该字段的值应设置为空字符串 `""`。**
5.  确保原始JSON的结构（所有的键和层级关系）在输出中被完整保留，只修改文本字段的内容。
6.  最终输出清洗和修正后的JSON对象字符串。

# **3. 清洗逻辑与规则 (Cleaning Logic & Rules)**

* **精确移除**：只移除明确表达"某事物不存在"的部分。如果一句话中既有对存在事物的描述，也有对不存在事物的描述（例如："道路右侧有建筑物，但未见行人。"），则应尝试只移除后半部分（例如，修改为："道路右侧有建筑物。"）。
* **保持积极描述的连贯性**：移除否定描述后，确保剩余的积极描述仍然通顺自然。可能需要进行非常细微的语序调整或连接词处理，但尽量保持原文风格。
* **字段值为空的处理**：如果一个原本有文本的字段，在移除了所有"描述缺失"的语句后，不再包含任何有效信息，则该字段的值必须更新为空字符串 `""`。
* **谨慎处理上下文**：
    * 如果一句"未见XX"是为了引出对可见度受限的描述（例如："高架桥结构及周围元素限制了对潜在行人区域的可视性，因此未见行人。"），优先移除"因此未见行人"部分，并审慎判断"限制了可视性"这类描述是否属于对当前环境的积极描述（例如"浓雾导致能见度低"是积极描述环境），还是仅仅是解释"未见"的原因。**本次任务主要聚焦于移除直接的"未见XX"陈述。**
    * **例外情况**：如果"未见异常"、"未见明显危险"这类表述是作为一个整体的、有意义的判断（通常表示情况正常或安全），则可以酌情保留。但本次任务主要针对具体物体或特征的"未见"。（如果难以判断，优先移除包含"未见"的直接陈述）。

# **4. 处理示例 (Processing Example)**

**输入JSON片段：**
```json
{
  "aspect_road_type_and_features": "此路段为沥青路面，路面干燥，但未见明显的硬路肩，也无中央隔离带。",
  "aspect_vulnerable_road_users_overview": "由于视野良好，可以确认在人行道及附近区域均未见行人或骑行者。",
  "aspect_traffic_control_elements": "前方有一个交通信号灯，当前为红灯。未见其他类型的交通标志。"
}
```

**期望输出JSON片段：**
```json
{
  "aspect_road_type_and_features": "此路段为沥青路面，路面干燥。",
  "aspect_vulnerable_road_users_overview": "",
  "aspect_traffic_control_elements": "前方有一个交通信号灯，当前为红灯。"
}
```

# **5. 输出格式要求 (Required Output Format)**

你的最终输出**必须且只能是**一个完整的、经过清洗和修正的JSON对象字符串。不要在JSON代码块前后添加任何解释性文字、注释或对话。

---

**任务开始：请处理我接下来提供的JSON字符串，并返回清洗后的版本。**

'''

import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import tempfile
import shutil
import gc
import signal
import sys
import time
import jsonlines

load_dotenv()


def process_single_file(args):
    """
    处理单个文件的函数
    :param args: 包含所有必要参数的元组 (json_path, image_path, api_key, temp_dir)
    """

    # 设置进程信号处理
    def signal_handler(signum, frame):
        print(f"进程收到终止信号: {signum}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    json_path, image_path, api_key, temp_dir = args

    try:
        client = genai.Client(api_key=api_key)

        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # 读取图片文件
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        model = "gemini-2.5-pro-preview-05-06"
        # model = "gemini-2.5-flash-preview-05-20"
        contents = [
            types.Content(
                role="user",
                parts=[
                    # types.Part.from_text(text=json.dumps(json_data, ensure_ascii=False)),
                    types.Part.from_text(text="""INSERT_INPUT_HERE"""),
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
                types.Part.from_text(text=prompt),
            ],
        )

        # 添加重试机制
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    config=generate_content_config,
                    contents=contents,
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"API调用失败，{retry_delay}秒后重试: {str(e)}")
                time.sleep(retry_delay)

        short_caption_list = json.loads(response.text)
        image_name = os.path.splitext(os.path.basename(json_path))[0]

        result = {
            "imageId": image_name,
            "short_caption_list": short_caption_list
        }

        # 将结果写入临时文件
        temp_file = os.path.join(temp_dir, f"{image_name}.json")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)

        # 清理内存
        del json_data
        del image_bytes
        del contents
        del response
        del short_caption_list
        gc.collect()

        return temp_file

    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        return None


def get_processed_images(output_file):
    """
    获取已经处理过的图片ID列表
    """
    processed_images = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_images.add(data['imageId'])
                except json.JSONDecodeError:
                    continue
    return processed_images


def generate(max_files=None, num_processes=None):
    """
    处理BDD100K数据集中的文件
    :param max_files: 要处理的最大文件数量，None表示处理所有文件
    :param num_processes: 使用的进程数量，None表示使用CPU核心数
    """
    # 设置路径
    json_dir = r"E:\playground\ai\datasets\bdd100k\100K\bdd100k_labels\bdd100k\labels\100k\train"
    image_dir = r"E:\playground\ai\datasets\bdd100k\100K\bdd100k_images\bdd100k\images\100k\train"
    output_file = "bdd100k_detailed_description.jsonl"
    image_ids_txt = "bdd100k_image_ids.txt"

    # 读取 imageId 列表
    with open(image_ids_txt, 'r', encoding='utf-8') as f:
        image_ids = [line.strip() for line in f if line.strip()]

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 获取已处理的图片ID
        processed_images = get_processed_images(output_file)
        print(f"已处理图片数量: {len(processed_images)}")

        # 只处理 image_ids_txt 中的 imageId
        image_ids = [img_id for img_id in image_ids if img_id not in processed_images]
        if max_files is not None:
            image_ids = image_ids[:max_files]
        total_files = len(image_ids)
        print(f"待处理文件数量: {total_files}")
        if total_files == 0:
            print("没有需要处理的文件")
            return

        # 准备处理参数
        api_key = os.environ.get("GEMINI_API_KEY")
        process_args = []
        for image_id in image_ids:
            json_path = os.path.join(json_dir, f"{image_id}.json")
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if os.path.exists(json_path) and os.path.exists(image_path):
                process_args.append((json_path, image_path, api_key, temp_dir))
            else:
                print(f"缺少文件: {json_path} 或 {image_path}")

        # 设置进程数
        if num_processes is None:
            num_processes = max(1, os.cpu_count() - 1)  # 保留一个CPU核心
        print(f"使用进程数: {num_processes}")

        # 使用ProcessPoolExecutor处理文件
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # 提交所有任务
            future_to_args = {
                executor.submit(process_single_file, args): args
                for args in process_args
            }

            # 使用tqdm显示进度
            with tqdm(total=len(process_args), desc="处理进度") as pbar:
                # 处理完成的任务
                for future in as_completed(future_to_args):
                    try:
                        temp_file = future.result()
                        if temp_file and os.path.exists(temp_file):
                            # 读取临时文件并写入最终输出
                            with open(temp_file, 'r', encoding='utf-8') as in_f:
                                result = json.load(in_f)
                                with open(output_file, 'a', encoding='utf-8') as out_f:
                                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                            # 删除临时文件
                            os.remove(temp_file)
                    except Exception as e:
                        print(f"处理结果时出错: {str(e)}")
                        # 记录错误信息到日志文件
                        with open("error_log.txt", "a", encoding="utf-8") as log_f:
                            log_f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {str(e)}\n")
                    finally:
                        pbar.update(1)
                        # 定期进行垃圾回收
                        if pbar.n % 5 == 0:  # 更频繁的垃圾回收
                            gc.collect()

        print(f"\n处理完成，结果已保存到: {output_file}")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_single(args):
    image_id, detailed_desc, api_key, temp_dir = args
    try:
        client = genai.Client(api_key=api_key)
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=json.dumps(detailed_desc, ensure_ascii=False)),
                ],
            )
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            system_instruction=[
                types.Part.from_text(text=prompt),
            ],
        )
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-pro-preview-05-06",
                    config=generate_content_config,
                    contents=contents,
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"API调用失败，{retry_delay}秒后重试: {str(e)}")
                time.sleep(retry_delay)
        cleaned_desc = json.loads(response.text)
        result = {
            "imageId": image_id,
            "cleaned_description": cleaned_desc
        }
        return result
    except Exception as e:
        print(f"处理imageId {image_id} 时出错: {str(e)}")
        return None


def process_jsonl_batch(input_jsonl, output_jsonl, api_key):
    """
    顺序处理JSONL文件，每行调用大模型清洗描述，结果写入新JSONL
    """
    from tqdm import tqdm
    import tempfile, shutil, gc

    # 读取所有待处理数据
    with jsonlines.open(input_jsonl, 'r') as reader:
        items = [line for line in reader]
    print(f"共需处理 {len(items)} 条数据")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        with jsonlines.open(output_jsonl, 'a') as writer:
            for item in tqdm(items, desc="清洗进度"):
                image_id = item.get('imageId')
                detailed_desc = item.get('short_caption_list')
                res = process_single((image_id, detailed_desc, api_key, temp_dir))
                if res:
                    writer.write(res)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    input_jsonl = "bdd100k_detailed_description.jsonl"
    output_jsonl = "remove_detailed_description_negative.jsonl"
    api_key = os.environ.get("GEMINI_API_KEY")
    process_jsonl_batch(input_jsonl, output_jsonl, api_key)
