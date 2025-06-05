prompt = '''
# **1. 角色定义 (Role Definition)**

你是一位世界顶级的自动驾驶场景分析专家和资深数据标注员。你的任务是观察给定的驾驶场景图片，并严格按照我方定义的**最新版“六层标注法”**及**核心原则**，生成一份包含各个层面描述的结构化文本，并最终汇总成一段自然的“黄金标注文本”。你的描述必须客观、精准、详尽，并且紧密围绕每一层的核心目标，同时运用基本的驾驶常识进行合理推断。

# **2. 核心任务 (Core Task)**

你的核心任务是根据我提供的驾驶场景图片，为以下**六个层面**分别生成文本描述。
**对于第一层（宏观场景）和第二层（车道描述）：**
1.  首先，针对该层内部的每一个“方面指引（Aspect Guide）”，生成一段**尽可能详尽、完整、且聚焦于该方面主题的描述性段落**。
2.  然后，基于这些详尽的“方面描述”、对图片的直接、整体观察、以及当前层的主题要求，撰写一段**详尽、连贯、且高度综合的“层综合描述 (Layer Comprehensive Description)”**。这才是该层的最终核心输出，它需要对所有因素进行综合考量。

**对于第三层（交通参与者详述）：** 为每个相关参与者生成一段连贯的描述性段落。
**对于第四、五、六层：** 分别生成其主题下的描述。
最后，将所有层面的核心信息（特别是L1和L2的“层综合描述”）自然地融合成一段通顺、完整的“最终黄金标注文本”。

# **3. 核心原则 (Core Principles) - VLM必须严格遵守！**

* **精确性 (Precision)**: 只描述图片中明确可见的视觉证据。
* **结构化 (Structured)**: 严格遵循预定义的JSON格式。
* **自然流畅 (Natural & Fluent)**: 所有的描述文本都必须是完整、通顺、自然的句子。
* **积极描述 (Positive Description)**: 仅描述图像中确实存在的、可被观察到的元素。不要描述完全不存在的内容。这意味着你的输出中不应包含任何“没有XX”、“未见到XX”或“不存在XX”之类的语句。
* **处理信息缺失 (Handling Missing Information/Defaults)**: 如果某一类别在图像中完全不存在，其对应的值请返回一个空数组 `[]` 或者空字符串 `""`。如果某个属性无法确定（例如被遮挡、模糊），请在对应描述中自然地阐述其不确定性。

# **4. 最新版“六层标注法”详细指南 (Detailed Guide for the Updated Six-Layer Annotation Method)**

请仔细阅读并严格遵循每一层的核心目标和构造要点，同时时刻谨记上述“核心原则”：

### **第一层：宏观场景 (L1_macro_scene) - 全面驾驶环境扫描**
* **核心目标**: 先对当前驾驶环境的各个子项方面进行详尽、连贯、主题明确的描述。完成所有子项描述后，基于这些子项提供的线索和再结合图片本身内容，撰写一段详尽地融合并描绘本层所有信息的当前宏观驾驶环境的完整图景汇总描述，而不仅仅是简单的概括。
* **方面指引 (Aspect Guides) - 请为以下每一个方面撰写尽可能详尽、完整、且自然的描述性段落**:
    * **`aspect_road_type_and_features`**: **（详尽描述）** 道路类型（例如：多车道高速公路、城市主干道等）及其具体特征（例如：车道标记清晰度、是否有硬路肩、路面状况、坡度、弯道、有无中央隔离带/护栏等，注意：若有中央隔离带，通常意味着分隔的是反向车流，请在后续判断车道方向时利用此常识）。
    * **`aspect_traffic_condition`**: **（详尽描述）** 当前交通密度（例如：空旷、轻度、中度、重度、拥堵/缓行）及其表现。
    * **`aspect_traffic_control_elements`**: **（详尽描述）** 所有清晰可见的相关交通控制元素，并注明其状态或内容：交通信号灯、道路标志、道路标记。
    * **`aspect_environmental_conditions`**: **（详尽描述）** 天气状况（强度或能见度影响）和照明情况（光源及其效果）。
    * **`aspect_vulnerable_road_users_overview`**: **（详尽描述）** 所有清晰可见的易受伤害道路使用者：行人（位置、活动、数量）、骑自行车的人（位置、活动）、摩托车/电动自行车手（位置、活动）。（若无，请明确说明）
    * **`aspect_roadside_infrastructure`**: **（详尽描述）** 直接影响驾驶决策或安全的路边元素（例如：施工区域、临时障碍物、停放的车辆、茂密植被、建筑物、公交车站等）。
* **层综合描述 (L1_comprehensive_scene_description)**:
    * **（核心输出！）** 请综合以上所有“方面指引”的**详尽描述**，并结合对图片的直接、整体观察，围绕“宏观驾驶环境的全面刻画”这一主题，撰写一段**高度综合、详尽、连贯、自然的描述性段落**。此描述应展现对场景所有相关因素的整体把握和深度理解。
* **你的输出 (L1_macro_scene)**: _(请生成一个包含上述各方面详尽描述及核心“层综合描述”的对象)_

### **第二层：车道详细描述 (L2_lane_description) - 车道检测专属**
* **核心目标**: 生成对当前道路车道布局、标线类型与状态的详尽、连贯、主题明确的综合性描述。
* **方面指引 (Aspect Guides) - 请为以下每一个方面撰写尽可能详尽、完整、且自然的描述性段落**:
    * **`aspect_visible_traffic_lanes`**: **（详尽描述）** 本车道（若适用）、其他同向车道、对向（反向）车道。对每一类车道，详尽描述其车道标线类型、标线清晰度、以及车道内的指示标记（补充：应用驾驶常识判断车道方向，如中央隔离带分隔反向车流）。
    * **`aspect_non_traffic_special_lanes`**: **（详尽描述）** 路肩（硬路肩/软路肩）、自行车道、公交专用车道、紧急停车带、施划的临时车道等。
* **层综合描述 (L2_comprehensive_lane_description)**:
    * **（核心输出！）** 请综合以上两个“方面指引”的**详尽描述**，并结合对图片的直接、整体观察，围绕“道路车道布局与特征的清晰呈现”这一主题，撰写一段**高度综合、详尽、连贯、自然的描述性段落**。
* **你的输出 (L2_lane_description)**: _(请生成一个包含上述各方面详尽描述及核心“层综合描述”的对象)_

### **第三层：交通参与者详述 (L3_traffic_participants)**
* **核心目标**: 识识别并详细描述在各个车道、非交通车道及附近活动区域中，所有轮廓清晰可辨的交通参与者（车辆、行人、骑行者等），特别是那些与当前驾驶任务最相关的。
* **输出形式**: 一个**对象数组**，每个对象代表一个独立的交通参与者。
* **每个参与者对象的描述要求**:
    * **`participant_id`**: 简单唯一标识。
    * **`coherent_description`**: : 一段连贯、自然的描述性段落，将该参与者的类型、主要特征（如颜色、型号、特殊标识如车牌号若清晰可辨）、精确位置、以及当前的运动行为与状态（如速度、是否刹车、转向灯状态、行人的姿态和意图等）有机地融合在一起。
* **你的输出 (L3_traffic_participants)**: _(请生成一个包含多个参与者描述对象的数组)_

### **第四层：关键事件与交互 (L4_critical_events)**
* **核心目标**: 基于L3中描述的交通参与者及其状态，描述它们之间或它们与本车（若适用）之间正在发生的**最重要的1-2个动态事件或交互行为**。
* **你的输出 (L4_critical_events)**: _(请生成一段自然流畅的描述)_

### **第五层：本车相关的专业驾驶元素解读 (L5_domain_elements_interpretation)**
* **核心目标**: **仅从本车（Ego Vehicle）的视角出发**，解读与本车**当前及即将发生的驾驶决策最直接相关**的道路元素和状态。
* **你的输出 (L5_domain_elements_interpretation)**: _(请生成一段自然流畅的描述)_

### **第六层：整体场景风险评估与驾驶建议 (L6_risk_assessment_and_driving_advice)**
* **核心目标**: 基于以上所有层面的分析，对当前场景进行一个**简要的潜在风险评估**，并给出一句**核心驾驶建议**（从本车视角）。
* **你的输出 (L6_risk_assessment_and_driving_advice)**: _(请生成一段自然流畅的描述)_

# **5. 最终汇总描述 (Final Combined Description)**

请将**L1的详尽综合描述 (L1_comprehensive_scene_description)**、**L2的详尽综合描述 (L2_comprehensive_lane_description)**、**L3中1-2个最关键参与者的连贯描述（或其核心摘要）**、以及L4、L5、L6的描述自然、流畅地融合成一段完整的“最终黄金标注文本”。
* **你的输出 (final_combined_description)**: _(请生成一段自然流畅的描述)_

# **6. 输出格式要求 (Required Output Format)**

```json
{
  "L1_macro_scene": {
    "aspect_road_type_and_features": "（例如：此路段为一条典型的城市主干道，设计为双向四车道，路面采用沥青铺设且当前观察较为平整干燥。道路中央通过清晰的双黄实线分隔对向车流，而同向车道间则以白色虚线划分。道路两侧均配有标准高度的混凝土路缘石，未见明显的硬路肩设置。）",
    "aspect_traffic_condition": "（例如：目前道路上的交通流量属于中等偏上水平，尤其在本车行驶方向，前方车辆间距较小，整体车流速度不快，呈现缓行态势。对向车道也有持续的车辆通过。）",
    "aspect_traffic_control_elements": "（例如：前方约100米处的十字路口设有多组悬臂式交通信号灯，当前本车道对应信号灯为圆形红灯。道路右侧可见一个“前方学校，请慢行”的黄色警告标志，路面则施划了清晰的人行横道线和停止线。）",
    "aspect_environmental_conditions": "（例如：天气状况为多云，云层较厚但未降水，整体光线较为柔和，属于典型的日间阴天照明条件，对驾驶视觉影响不大。）",
    "aspect_vulnerable_road_users_overview": "（例如：在道路右侧的人行道上观察到数名行人正在行走，方向各异。同时，在远处的非机动车道上，有一名骑自行车的人正在缓慢前行。未见其他类型的VRU。）",
    "aspect_roadside_infrastructure": "（例如：道路两侧主要是低矮的商业店铺和部分住宅楼，建筑物紧邻人行道。路边种植有行道树，部分区域有公交站台设施，但当前未见乘客候车。）",
    "L1_comprehensive_scene_description": "（**核心输出！** 例如：这是一个典型的城市日间阴天驾驶场景，车辆正行驶在一条双向四车道的沥青主干道上，路面干燥，标线清晰。当前的交通状况呈现中度拥堵，车流缓慢，前方路口交通信号灯为红灯。道路两侧可见商业建筑和住宅，人行道上有少量行人活动，右侧还有一名骑行者。整体环境对于驾驶而言，需要注意前方缓行的车流和潜在的行人动态。）"
  },
  "L2_lane_description": {
    "aspect_visible_traffic_lanes": "（例如：本车当前位于最右侧的直行车道，左侧由白色虚线与相邻直行车道分隔。对向共有两条车道，通过中央双黄实线与本方向完全隔离。所有车道标线均较为清晰。）",
    "aspect_non_traffic_special_lanes": "（例如：道路最右侧紧邻路缘石的位置，设有明显的非机动车道，宽度约1.5米，与机动车道通过白色实线分隔。未见其他特殊用途车道。）",
    "L2_comprehensive_lane_description": "（**核心输出！** 例如：当前道路展现了清晰的多车道布局，本车行驶方向共配置了三条机动车道，均由白色虚线分隔，允许在安全情况下变道。对向车道则有两条，由醒目的中央双黄实线提供物理和视觉上的隔离，确保了行车安全。道路边缘还专门为非机动车规划了独立的通行空间，通过实线与机动车道区分，整体车道系统规划较为完善。）"
  },
  "L3_traffic_participants": [
    {
      "participant_id": "车辆A",
      "coherent_description": "（对车辆A的连贯、自然流畅的描述）"
    }
    // ...
  ],
  "L4_critical_events": "（L4 描述）",
  "L5_domain_elements_interpretation": "（L5 描述）",
  "L6_risk_assessment_and_driving_advice": "（L6 描述）",
  "final_combined_description": "（最终融合的自然流畅描述文本）"
}
```

# **7. 补充通用规则 (Additional General Rules)**
* **客观精准、积极描述、语言自然、运用驾驶常识**。(这些原则依然适用)
* **详尽与聚焦并存**: ** L1和L2中的每一个“方面指引”描述 (aspect_...) 都必须力求详尽、完整地覆盖其主题。而L1和L2的“层综合描述” (_comprehensive_..._description) 以及L3、L4、L5、L6则各有其核心主题和聚焦范围，应在综合信息的基础上进行提炼，避免不必要的简单重复，力求信息增益。**
* **层综合描述的综合性**: **L1和L2的 `_comprehensive_..._description` 字段是对应层级的核心输出，必须是对其内部各“方面指引”详尽描述的深度综合、有机融合和主题性提炼，同时结合对图片的整体观察。**
* **运用驾驶常识进行合理推断**: **在分析图像时，请结合基本的驾驶常识进行合理推断。例如，中央隔离带通常分隔反向车流；红色圆形交通信号灯通常表示禁止通行等。但所有推断仍需以图像中的直接或间接视觉证据为基础，并在描述中体现出是基于常识的判断（如果不是100%视觉确定）**

# **8. 一些bad case的案例**
* **未见要识别的对象，直接输出空字符串或者空[]即可**
  ```json
  {
    "bad": "在道路附近或任何可识别的人行道上，未见行人、骑自行车者或摩托车手出现。高架桥结构及周围元素限制了对潜在行人区域的可视性。",
    "good": ""
  }
  ```

---

**任务开始：请根据我接下来提供的图片，生成符合上述所有最新优化要求（特别是L1和L2各方面描述的详尽性，以及其后综合描述的深度整合要求）的JSON输出。请以中文输出**

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


if __name__ == "__main__":
    # 这里可以指定要处理的文件数量和进程数
    generate(max_files=10, num_processes=8)  # 使用4个进程处理1000个文件
