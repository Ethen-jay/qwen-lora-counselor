# 心理情感分析模型微调（Qwen2.5-7B-Instruct + LoRA SFT）

通用大模型在心理陪伴场景中回答偏百科、爱讲道理、缺乏共情。本项目基于
**PsyDTCorpus（心理诊疗对话语料）** 对 Qwen2.5-7B-Instruct 做领域 LoRA 微调，
使其先共情接纳、再温和引导。

## 数据处理流程

```
官方原始数据（messages 格式，自带 train/test 划分）
   │
   ├─① 01_to_sharegpt.py   原始 messages → ShareGPT（system/human/gpt）
   ├─② 02_to_alpaca.py     ShareGPT → Alpaca（instruction/input/output）
   ├─③ 03_clean.py         四层清洗：格式校验 → 文本清洗 → 长度过滤 → MD5 去重
   └─④ 04_token_stats.py   token 长度分布统计（支撑长度阈值选择）
```

## 数据规模

| 项目 | 数值 |
|---|---|
| 最终训练样本 | **4760 条**多轮对话 |
| 官方划分 | train（多轮 packing）/ test（单轮 split） |
| 清洗策略 | 格式校验 → 文本清洗 → 长度过滤 → MD5 去重 |

## 微调配置（LLaMA-Factory）

- 方法：LoRA + SFT，rank=8、alpha=32，覆盖全部线性层
- 可训练参数：约 0.5%
- 学习率 2e-5，Cosine + warmup，2 epochs，bf16，cutoff_len 2048

## 结果

- 训练 loss 由 **2.65 稳定降至 1.33**，全程无发散（见 loss_curve.png）
- 应答风格由"直接给建议、快速说教"明显转为"先共情接纳、再温和引导"
- 训练后将 adapter 与原模型合并为完整模型，以 vLLM 部署为 OpenAI 兼容接口，
  跑通"数据处理 → 微调 → 合并 → 部署"完整链路

> 注：数据文件为第三方语料（PsyDTCorpus），受版权限制未随仓库分发，
> 仓库仅包含数据处理脚本。
