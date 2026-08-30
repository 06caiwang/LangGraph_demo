from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户的身份给予合适的方案"""
    user_role = request.runtime.context.get("user_role", "初学者")
    base_prompt = "你是一名乐于助人的专家"

    if user_role == "初学者":
        return f"{base_prompt}, 用简单易懂的语言描述概念，避免使用专业术语"
    elif user_role == "专家":
        return f"{base_prompt}, 尽量提供详尽的技术解答，需要使用专业术语"
    return base_prompt

agent = create_agent(
    model="deepseek-v4-flash",
    context_schema=Context,
    middleware=[user_role_prompt],
)

# response = agent.invoke(
#     {
#         "messages": [{
#             "role": "user",
#             "content": "解释一下人工智能"
#         }],
#     },
#     context={"user_role": "初学者"},
# )
#
# print(response["messages"][-1].content)

# “人工智能”听起来很高大上，其实你可以把它想象成**给机器装上了一个“会学习的脑子”**。
#
# 不过这个脑子不是天生的，而是靠人类“喂”给它大量数据，让它自己找规律练出来的。
#
# 举个简单的例子：
#
# - 你教一个小孩认识猫，会指着真猫、图片说“这是猫”。
# - 人工智能也一样，我们给电脑看成千上万张猫的图片，它自己慢慢琢磨出“猫”和“狗”有什么不同。看多了以后，你再给它看一张新照片，它就能认出来“这是猫”。
#
# 所以，人工智能就是**让电脑通过“看经验”来学会做某件事**，而不是我们一条一条给它写死规则。
#
# 它的厉害之处在于：**它能从海量信息里找出人类很难发现的规律**，然后帮我们干活。比如：
#
# - 你手机里的语音助手，能听懂你说的话，就是因为它“听过”无数人说话。
# - 短视频推荐你喜欢看的，是因为它“观察”过你之前点了什么。
# - 自动驾驶汽车能避开行人，是因为它“见识”过无数种路况。
#
# 简单总结：人工智能 = **机器 + 大数据 + 会自己找规律 + 帮你完成任务**。
#
# 它不神秘，就像一个特别勤奋、见多识广的“数字实习生”，只不过这个实习生的“大脑”是电脑程序。

response = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "解释一下人工智能"
        }],
    },
    context={"user_role": "专家"},
)

print(response["messages"][-1].content)

"""
---

## 一、核心定义与基本范式

人工智能并非单一技术，而是包含多种范式与子领域的交叉学科。其经典定义可归为四类：

- **像人一样思考**：认知科学取向，强调建模人类认知过程（如知识表示、推理）。
- **像人一样行动**：图灵测试的实践路径，关注行为上的类人性。
- **理性地思考**：逻辑主义范式，基于形式化逻辑与演绎推理。
- **理性地行动**：智能体范式，追求在不确定环境中做出最优决策（即理性Agent）。

现代AI主流属于**理性行动**范式，以“智能体”为核心抽象：感知（Perception）→ 推理（Reasoning）→ 行动（Action），并通过学习（Learning）持续优化策略。

---

## 二、主要技术分支

### 1. 机器学习（Machine Learning）
机器学习是当前AI的核心驱动力，它使系统无需显式编程即可从数据中自动发现规律。按照学习范式可分为：

- **监督学习**（Supervised Learning）：利用标注数据学习映射函数，典型任务包括分类（Classification）与回归（Regression）。
- **无监督学习**（Unsupervised Learning）：从未标注数据中发现隐含结构，如聚类（Clustering）、降维（Dimensionality Reduction）。
- **强化学习**（Reinforcement Learning）：智能体通过与环境交互，依据奖励信号（Reward Signal）学习最优策略（Policy），适用于博弈、机器人控制等序贯决策场景。

### 2. 深度学习（Deep Learning）
基于多层人工神经网络（Artificial Neural Network, ANN）的表示学习方法。深度学习通过端到端的梯度反向传播（Backpropagation）自动提取多层次抽象特征，有效解决了传统特征工程的瓶颈。关键架构包括：

- **卷积神经网络（CNN）**：擅长处理网格结构数据，如图像、视频。
- **循环神经网络（RNN）/长短期记忆网络（LSTM）**：处理序列数据，如语音、文本。
- **Transformer**：基于自注意力机制（Self-Attention），已成为自然语言处理与大模型的主流架构，如GPT、BERT。

### 3. 自然语言处理（NLP）
使计算机理解、生成和交互人类语言。核心技术包括词嵌入（Word Embedding）、注意力机制（Attention）、预训练语言模型（Pretrained Language Model, PLM）等。代表性任务：机器翻译、文本摘要、情感分析、问答系统。

### 4. 计算机视觉（CV）
赋予机器“看”的能力，包括图像分类、目标检测（Object Detection）、语义分割（Semantic Segmentation）、图像生成（如扩散模型，Diffusion Models）等。

### 5. 知识表示与推理（Knowledge Representation & Reasoning）
将领域知识形式化并实现逻辑推理，包括本体论（Ontology）、知识图谱（Knowledge Graph）、描述逻辑（Description Logic）等，用于构建可解释的专家系统。

### 6. 机器人学（Robotics）
将感知、规划、控制融于物理实体，实现与物理世界的交互，涉及同时定位与地图构建（SLAM）、运动规划、人机协作等。

---

## 三、关键技术原理

- **神经网络的训练**：基于损失函数（Loss Function）最小化，使用优化器（如SGD、Adam）进行梯度下降。
- **泛化能力**：模型对未见过数据的适应能力，需通过正则化（Regularization）、交叉验证（Cross-Validation）等手段控制过拟合（Overfitting）。
- **表征学习**：将原始数据变换为机器可用的特征向量，深度学习通过隐层自动完成。
- **生成式模型**：如变分自编码器（VAE）、生成对抗网络（GAN）、扩散模型，用于生成新数据。

---

## 四、典型应用领域

| 领域 | 应用实例 |
|------|----------|
| 医疗健康 | 医学影像辅助诊断、药物分子设计、基因组学分析 |
| 金融科技 | 智能风控、量化交易、反欺诈 |
| 智能制造 | 预测性维护、质量检测、智能排产 |
| 自动驾驶 | 环境感知、路径规划、决策控制 |
| 智能客服 | 对话系统、意图识别、情感分析 |
| 教育 | 自适应学习、自动批改 |
| 安防 | 人脸识别、行为分析 |
| 科学发现 | 蛋白质结构预测（如AlphaFold）、材料设计 |

---

## 五、挑战与前沿议题

1. **可解释性（Explainability）**：深度模型常被视为“黑箱”，需发展可解释AI（XAI）技术，建立对决策过程的信任。
2. **数据与偏见**：训练数据可能包含社会偏见，导致模型产生歧视性输出；数据隐私与合规问题（如GDPR）亟待解决。
3. **鲁棒性与安全性**：对抗样本（Adversarial Examples）可轻易欺骗模型，需增强模型对恶意攻击的抵抗力及部署安全。
4. **常识与因果推理**：目前AI缺乏真正的常识推理与因果理解能力，大多依赖统计相关性。
5. **对齐问题（Alignment Problem）**：确保人工智能目标与人类价值观一致，避免潜在风险。
6. **资源消耗**：训练大模型能耗极高，需探索绿色算法与高效硬件。

---

## 六、未来发展方向

- **多模态AI**：融合视觉、语音、文本、传感等多通道信息，实现更接近人类的综合认知。
- **具身智能（Embodied AI）**：将AI嵌入物理系统，机器人与环境深度交互。
- **大模型与AGI**：从海量数据中涌现通用能力，逐步迈向通用人工智能（Artificial General Intelligence）。
- **神经符号融合**：将深度学习的感知能力与符号逻辑的推理能力结合，提升可解释性。
- **联邦学习与隐私计算**：在数据不出域的前提下实现多方协作训练。

---

人工智能不仅是一项技术，更是一场范式革命，它正在重塑人类社会的生产与生活模式。然而，其发展也伴随着伦理、法律与社会层面的复杂挑战。未来的AI将走向更强大的能力、更可靠的安全保障、更公平的价值分配，这需要计算机科学、认知科学、哲学与社会学等多学科协同推进。

"""
