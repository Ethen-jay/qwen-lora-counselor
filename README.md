# 情绪陪伴助手 · LoRA 微调

基于 **Qwen2.5-7B-Instruct** 的 LoRA 微调项目，让通用大模型学会心理咨询师的软语气，解决多轮对话中 Prompt 风格漂移和 RAG 无法生成"语气"的结构性短板。

## 为什么用 LoRA

| 方案 | 问题 |
|---|---|
| 纯 Prompt 约束 | 多轮对话 3 轮后风格漂移 |
| RAG | 只能管"回答什么"，管不了"怎么说"的语气 |
| 全参微调 | 7B bf16 全参约 14G，训练需 84G+ 显存，A800 单卡跑不起 |
| **LoRA** | 冻结原权重只训旁路低秩矩阵（约 3.5M 参数，占 0.5%），24G 显存可跑，推理时可合并回原权重无延迟 |

## 微调配置

| 参数 | 值 | 调优过程 |
|---|---|---|
| 基座模型 | Qwen2.5-7B-Instruct (Apache 2.0) | 对比过 Llama-3-8B、InternLM-7B，千问中文共情表达顶流 |
| LoRA rank | 8 | 默认 4 收敛慢风格改不动，升到 8 后效果显现 |
| LoRA alpha | 32 | 缩放系数 alpha/rank=4，让旁路话语权足够 |
| 覆盖层 | 42 层全线性（q/k/v/o/gate/up/down_proj） | 标准配置 |
| 可训练参数 | 约 3.5M（占总参数 0.5%） | 冻结原权重避免灾难性遗忘 |
| 数据集 | 灵心 4760 轮心理健康长对话 | ShareGPT → Alpaca 格式转换 |
| 清洗流程 | 格式校验 → 文本清洗 → 长度过滤 → MD5 去重（四层） | 开源数据需自行清理 |
| 优化器 | AdamW，Cosine scheduler + warmup | — |
| 学习率 | 2e-5 | 默认 1e-4 loss 震荡，降到 2e-5 平滑 |
| epochs | 2 | 3 轮后开始过拟合重复话术 |
| cutoff | 2048 | 默认 1024 多轮后半段被截断 |
| 精度 | bf16 | 7B 模型推荐精度 |
| Loss | 2.65 → 1.33，全程无震荡无发散 | — |

## 硬件

- GPU：A800 单卡 24G
- 训练框架：LLaMA-Factory（底层 Hugging Face Transformers）

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/Ethen-jay/qwen-lora-counselor.git
cd qwen-lora-counselor

# 2. 准备数据（灵心数据集，需自行下载并运行转换脚本）
# finetune/01_to_sharegpt.py → finetune/02_to_alpaca.py → finetune/03_clean.py

# 3. 训练（需自行安装 LLaMA-Factory）
# llamafactory-cli train your_config.yaml

# 4. 合并 LoRA 权重回原模型
# llamafactory-cli export --adapter_name your_lora --export_dir merged_model

# 5. 推理（vLLM 部署）
# python -m vllm.entrypoints.openai.api_server --model merged_model
```

## 项目结构

```
qwen-lora-counselor/
├── README.md                    # 本文档
├── .gitignore                   # 排除模型权重与密钥
├── finetune/                    # 数据预处理脚本
│   ├── 01_to_sharegpt.py       # 灵心 → ShareGPT 格式
│   ├── 02_to_alpaca.py         # ShareGPT → Alpaca 格式
│   ├── 03_clean.py             # 四层清洗
│   ├── 04_token_stats.py       # token 统计
│   ├── loss_curve.png          # 训练 Loss 曲线
│   └── README.md               # 脚本说明
├── eval_finetune.py             # 自动评测脚本
└── eval_finetune_qualitative.py # 定性风格对比评测
```

## 评估方法

风格是主观指标，无法用 BLEU 这类自动指标量化。项目采用：

1. **裁判模型三维打分**（情绪承接 / 语气温柔 / 多轮一致性，1-5 分）+ 中文情感分类模型辅助
2. **原版 vs 微调版双模型盲评**，打乱顺序让裁判模型打分
3. **定性观察**：连续 5 轮对话后风格稳定性、生硬祈使词占比变化
4. **过拟合观察**：epochs 3 之后 Loss 继续降但生成内容开始重复安慰话术，及时停在 2 轮

## 对比三种方案（同一测试用例）

| 方案 | 第 1 轮 | 第 3 轮 | 风格稳定性 |
|---|---|---|---|
| 通用千问 7B | 冷静理性："我给你几个建议" | 同上 | ✅ 但没共情 |
| Prompt 约束版 | 软语："我能感受到这有多难" | 回到冷静理性 | ❌ 3 轮后漂移 |
| **LoRA 微调版** | 软语："我能感受到这有多难" | 保持软语 | ✅ 不漂移 |

## License

训练代码 MIT，基座模型 Qwen2.5-7B-Instruct Apache 2.0，LoRA 权重需后续单独发布。
