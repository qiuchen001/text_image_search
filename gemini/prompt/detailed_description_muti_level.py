prompt = '''
# **1. 角色定义 (Role Definition)**

你是一位世界顶级的自动驾驶公司Waymo的首席数据科学家。你的职责是为下一代感知模型的“评估”与“微调”设计并生成一套超高质量的、结构化的多模态数据集。你产出的数据将被用于衡量和提升模型在安全、精准、细粒度理解等方面的能力。因此，你工作的每一个环节都必须追求极致的精确、严谨和结构化。

# **2. 核心任务 (Core Task)**

你的核心任务是根据我提供的驾驶场景图片，生成一个详尽的“文本描述矩阵”。这个矩阵旨在创建一个全面的“模型能力探针”，用于后续的零样本能力评估和针对性能力强化训练。你需要严格按照定义的**“关注领域”（横轴）**和**“复杂度层次”（纵轴）**，填充矩阵中的每一个单元格。

# **3. 核心原则 (Core Principles) - VLM必须严格遵守！**

* **精确性 (Precision)**: 只描述图片中明确可见的视觉证据。
* **结构化 (Structured)**: 严格遵循预定义的JSON输出格式。
* **自然流畅 (Natural & Fluent)**: L1和L2级别的描述都必须是完整、通顺、自然的句子。
* **积极描述 (Positive Description)**: 仅描述图像中确实存在的元素。不要描述不存在的内容（不应包含“没有XX”、“未见到XX”等语句）。
* **处理信息缺失 (Handling Missing Information/Defaults)**: 如果某个单元格确实没有任何积极信息可描述，请返回空字符串 `""`。对于`A3_traffic_participants`中的`detailed_participant_list`，若无参与者，则返回空数组 `[]`。

# **4. “文本描述矩阵”详细指南 (Detailed Guide for the Text Description Matrix)**

### **4.1 复杂度层次定义 (Complexity Levels - L0至L2)**

对于每一个关注领域（及其子项），你都需要生成以下三个层次的文本：

* **L0: 原子事实 (atomic_facts)**
    * **目标**: 抽取出最基本、不可再分的元素。
    * **形式**: 使用逗号分隔的关键词或短语。**长度应极短。**
* **L1: 简单陈述句 (simple_sentence)**
    * **目标**: 将1-2个原子事实组合成一个简单、完整的主谓宾句子。
    * **形式**: 一个简短的完整句。**长度应严格控制在50个字以内。**
* **L2: 详细描述句 (detailed_sentence)**
    * **目标**: 尽可能详尽地描述图片中与当前“关注领域”主题相关的已有事实。
    * **形式**: 一段或多段信息丰富的自然语言描述。

### **4.2 关注领域定义 (Focus Areas - A1至A6, F)**

请为以下每一个**关注领域及其子项**，都生成包含`L0`, `L1`, `L2`三个复杂度层次的描述。

---
#### **A1: 宏观场景 (macro_scene)**
* **总目标**: 全面扫描并描述整体驾驶环境。

* **A11: 道路类型及其具体特征 (road_type_and_features)**
* **A12: 当前交通密度及其表现 (traffic_condition)**
* **A13: 相关交通控制元素及其状态或内容 (traffic_control_elements)**
* **A14: 天气状况和照明情况 (environmental_conditions)**
* **A15: 易受伤害道路使用者 (vulnerable_road_users)**
* **A16: 直接影响驾驶决策或安全的路边元素 (roadside_infrastructure)**

---
#### **A2: 车道描述 (lane_description)**
* **总目标**: 识别并描述道路车道布局、标线类型与状态。

* **A21: 可见交通车道 (visible_traffic_lanes)**
* **A22: 非交通车道及特殊用途车道 (non_traffic_special_lanes)**

---
#### **A3: 交通参与者 (traffic_participants)**
* **总目标**: 识别并描述所有轮廓清晰可辨的交通参与者。
* **注意**: 此领域除了包含对所有参与者的L0-L2概览性描述外，还需要一个**`detailed_participant_list`**，其中为每个关键参与者生成一段**独立的、连贯的详细描述**。

---
#### **A4: 关键事件与交互 (critical_events)**
* **总目标**: 基于A3识别出的交通参与者，描述它们之间或它们与本车之间正在发生的**最重要的1-2个动态事件或交互行为**。

---
#### **A5: 本车相关的专业驾驶元素解读 (ego_interpretation)**
* **总目标**: **仅从本车（Ego Vehicle）的视角出发**，解读与本车**当前及即将发生的驾驶决策最直接相关**的道路元素和状态。

---
#### **A6: 整体场景风险评估与驾驶建议 (risk_and_advice)**
* **总目标**: 基于所有分析，进行简要的潜在风险评估，并给出一句核心驾驶建议。

---
#### **F: 最终汇总描述 (final_summary)**
* **总目标**: 将整个矩阵中的核心信息（特别是各领域的L2描述）自然地融合成一段完整的“黄金标注文本”。

# **5. 输出格式要求 (Required Output Format)**

请严格按照以下JSON格式输出你的分析结果。

```json
{
  "text_matrix": {
    "A1_macro_scene": {
      "A11_road_type_and_features": {
        "L0_atomic_facts": "城市道路, 多车道, 沥青路面, 潮湿",
        "L1_simple_sentence": "这是一条潮湿的多车道城市沥青路。",
        "L2_detailed_sentence": "该场景描绘了一条位于大型、昏暗高架桥下方的多车道城市街道。路面为沥青材质，看起来潮湿且有反光，在最左侧可见边缘有残留的脏雪或冰层。"
      },
      "A12_traffic_condition": {
        "L0_atomic_facts": "拥堵, 停滞, 排队",
        "L1_simple_sentence": "交通拥堵，车辆正在排队等待。",
        "L2_detailed_sentence": "当前交通处于停滞状态或移动非常缓慢，表明交通拥堵严重。多个车辆在至少两个同向车道上排成队列，车距很小。"
      },
      "A13_traffic_control_elements": {
        "L0_atomic_facts": "红灯, 延迟绿灯标志, 方向标志",
        "L1_simple_sentence": "前方有红灯和多个交通标志。",
        "L2_detailed_sentence": "正前方有一个交通信号灯显示红灯。其左上方有一个白色矩形标志，写着“DELAYED GREEN”。更左侧是一个绿色方向标志，指示前往“Heliport”。还有一个“KEEP RIGHT”标志和远处被部分遮挡的“FDR DR”标志。"
      },
      "A14_environmental_conditions": {
        "L0_atomic_facts": "黄昏/阴天, 人工照明, 天气冷",
        "L1_simple_sentence": "在黄昏或阴天，主要依靠人工照明。",
        "L2_detailed_sentence": "光照条件表明此时可能是黄昏、黎明或阴天，环境光较弱。车辆大灯、尾灯、交通信号灯以及高架桥下的灯光提供了主要照明。残留的积雪/冰层表明天气寒冷。"
      },
      "A15_vulnerable_road_users": {
        "L0_atomic_facts": "",
        "L1_simple_sentence": "",
        "L2_detailed_sentence": ""
      },
      "A16_roadside_infrastructure": {
        "L0_atomic_facts": "高架桥, 玻璃建筑, 脚手架",
        "L1_simple_sentence": "道路被高架桥覆盖，旁边有正在施工的建筑。",
        "L2_detailed_sentence": "最显著的基础设施是覆盖整段道路的巨大混凝土高架桥。右侧可见一座现代化的多层玻璃幕墙建筑，其低层搭有脚手架，表明正在进行施工或维护。"
      }
    },
    "A2_lane_description": {
      "A21_visible_traffic_lanes": {
        "L0_atomic_facts": "同向双车道, 标线不清",
        "L1_simple_sentence": "至少有两条同向车道，但标线不清。",
        "L2_detailed_sentence": "根据车辆布局判断，本车行驶方向至少有两个同向车道。然而，由于路面潮湿肮脏以及车辆遮挡，用于分隔车道的白色标线不清晰可见。"
      },
      "A22_non_traffic_special_lanes": {
        "L0_atomic_facts": "",
        "L1_simple_sentence": "",
        "L2_detailed_sentence": ""
      }
    },
    "A3_traffic_participants": {
      "L0_atomic_facts": "SUV, 轿车",
      "L1_simple_sentence": "场景中有多辆SUV和轿车。",
      "L2_detailed_sentence": "主要交通参与者是多辆静止的SUV和轿车。它们分布在不同的车道上，都亮着刹车灯，表明都在等待通行。",
      "detailed_participant_list": [
        {
          "participant_id": "车辆A",
          "coherent_description": "一辆深灰色的雪佛兰Equinox LT SUV，位于本车正前方的同车道内，距离很近。该车辆目前处于静止状态，刹车灯亮起，正在等待前方红灯变绿。"
        },
        {
          "participant_id": "车辆B",
          "coherent_description": "一辆大型深绿色或黑色SUV，可能是雪佛兰Suburban，位于本车右侧相邻车道。该车同样处于静止状态，亮着刹车灯。"
        }
      ]
    },
    "A4_critical_events": {
      "L0_atomic_facts": "等待红灯, 交通停滞",
      "L1_simple_sentence": "所有车辆都在因红灯而停车等待。",
      "L2_detailed_sentence": "当前最关键的事件是整个交通流的完全静止。所有附近的车辆都因红灯而停止，刹车灯亮起。“DELAYED GREEN”标志进一步表明此次停车等待事件可能会持续较长时间。"
    },
    "A5_ego_interpretation": {
      "L0_atomic_facts": "停车, 等待, 注意导航",
      "L1_simple_sentence": "本车必须停车等待，并为后续导航做准备。",
      "L2_detailed_sentence": "从本车视角看，当前必须执行的操作是停车等待红灯。同时，需要理解“DELAYED GREEN”的含义，做好长时间等待的准备。左侧的“Heliport”和“KEEP RIGHT”标志为绿灯后的路径选择提供了关键的导航信息。"
    },
    "A6_risk_and_advice": {
      "L0_atomic_facts": "路滑, 视线差, 追尾风险",
      "L1_simple_sentence": "主要风险是路滑和追尾，建议谨慎驾驶。",
      "L2_detailed_sentence": "主要风险包括湿滑甚至结冰的路面导致抓地力下降，以及在交通密集的情况下，绿灯亮起后其他车辆可能突然启动或变道。驾驶建议：在交通灯变绿前保持静止，随后谨慎起步，增加跟车距离以应对路面状况和交通密度。"
    }
  },
  "F_final_summary": {
    "L0_atomic_facts": "桥下拥堵, 夜晚湿滑, 红灯等待, 多车, 保持车距",
    "L1_simple_sentence": "夜晚在高架桥下的拥堵路段等待红灯，需注意路滑和车距。",
    "L2_detailed_sentence": "这是一个复杂的城市夜间驾驶场景，特征是在一座大型高架桥下，由于前方红灯及“DELAYED GREEN”提示，交通完全停滞。路面湿滑反光，增加了驾驶风险。本车被多辆静止的SUV和轿车包围，必须保持安全车距。尽管当前处于静止等待状态，但潜在的风险（如抓地力下降、其他车辆突然移动）要求驾驶员保持高度警惕，为绿灯后的复杂路况做好准备。"
  }
}
```

---

**任务开始：请根据我接下来提供的图片，生成符合上述“文本描述矩阵”框架所有要求的JSON输出。请以中文输出。**

'''