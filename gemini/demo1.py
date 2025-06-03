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

        # model = "gemini-2.5-pro-preview-05-06"
        model = "gemini-2.5-flash-preview-05-20"
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
    2. 你的所有输出必须严格来自于已标注的数据，给你提供图片只是为了让你可以更好的描述已标注的数据
    3. 尽量不要出现否定词，以下举例一些好的案例和不好的案例对比：
    [
    {
      \"bad_case\":  \"汽车交通灯信息无\"
      \"good_case\":  \"\" # bad_case采用了否定词，既然没有交通灯信息就不要写了
    },
    {
      \"bad_case\":  \"前方有未遮挡行人\"
      \"good_case\":  \"前方有行人\" # bad_case使用了词汇：未遮挡，这在人类正常描述中不太常见，描述太多书面化，机械化，而且还使用了否定词
    }
    
    ]
    
    """),
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
    output_file = "bdd100k_captions.jsonl"

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 获取已处理的图片ID
        processed_images = get_processed_images(output_file)
        print(f"已处理图片数量: {len(processed_images)}")

        # 获取所有JSON文件
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        # 过滤掉已处理的文件（统一使用不带后缀的文件名进行比较）
        json_files = [f for f in json_files if os.path.splitext(f)[0] not in processed_images]
        
        # 如果指定了最大文件数，则限制处理数量
        if max_files is not None:
            json_files = json_files[:max_files]
        
        total_files = len(json_files)
        print(f"待处理文件数量: {total_files}")

        if total_files == 0:
            print("没有需要处理的文件")
            return

        # 准备处理参数
        api_key = os.environ.get("GEMINI_API_KEY")
        process_args = []
        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            image_name = os.path.splitext(json_file)[0]
            image_path = os.path.join(image_dir, f"{image_name}.jpg")
            if os.path.exists(image_path):
                process_args.append((json_path, image_path, api_key, temp_dir))

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


if __name__ == "__main__":
    # 这里可以指定要处理的文件数量和进程数
    generate(max_files=1000, num_processes=8)  # 使用4个进程处理1000个文件
