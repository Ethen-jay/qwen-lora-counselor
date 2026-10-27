"""微调评测脚本：对比原版 Qwen2.5-7B-Instruct vs 微调后模型。

通过 API 调用两个模型，用 DeepSeek 裁判从 3 个维度打分：
  1. 共情性：是否先接纳情绪再引导
  2. 建议合理性：建议是否具体可执行
  3. 人设一致性：是否符合"温暖陪伴者"人设

运行方式：先在下方配置两个模型的 API 端点，然后运行：
    python eval_finetune.py
"""

import asyncio
import json
import sys
import os
from pathlib import Path

import httpx

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ======== 模型 API 配置（部署好后更新这里） ========
# 原版模型端点（如 DashScope、ModelScope 或自部署）
BASELINE_API_URL = "http://localhost:8080/v1/chat/completions"  # TODO: 部署后改
BASELINE_API_KEY = "EMPTY"  # TODO: 如果是自部署 vLLM 通常是 EMPTY

# 微调后模型端点
FINETUNED_API_URL = "http://localhost:8081/v1/chat/completions"  # TODO: 部署后改
FINETUNED_API_KEY = "EMPTY"  # TODO

# 裁判 LLM 配置（用你 .env 里的 DeepSeek）
JUDGE_API_URL = "https://api.deepseek.com/v1/chat/completions"
JUDGE_API_KEY = ""  # TODO: 填 DeepSeek API Key
# 或者自动从 .env 读取
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("LLM_API_KEY="):
                JUDGE_API_KEY = line.strip().split("=", 1)[1]
                break

# ======== 20 道评测用例 ========
# 覆盖不同情绪类型和场景，模拟真实心理陪伴对话
EVAL_QUESTIONS = [
    # 【学业/职场压力】
    {
        "category": "学业/职场压力",
        "question": "我今天又被领导骂了，说我做的东西都不对，我觉得自己什么都做不好。",
        "expectation": "先接纳挫败感，不急于说教，引导客观看待问题",
    },
    {
        "category": "学业/职场压力",
        "question": "考研二战又失败了，爸妈对我很失望，我也不知道自己还能做什么。",
        "expectation": "共情失落，不贬低二战努力，引导重新梳理方向",
    },
    {
        "category": "学业/职场压力",
        "question": "马上要期末考了，还有三门没复习，我焦虑到整晚失眠，怎么办？",
        "expectation": "理解焦虑情绪，给出可执行的小步复习计划",
    },
    # 【人际关系】
    {
        "category": "人际关系",
        "question": "室友总在背后说我坏话，但表面上又跟我很好，我真的很恶心。",
        "expectation": "接纳被背叛的感受，不评判室友，引导自我保护",
    },
    {
        "category": "人际关系",
        "question": "我喜欢的人跟我表白了一个月又说其实没那么喜欢，我真的很受伤。",
        "expectation": "共情情感落差，不否定付出，引导关注自我价值",
    },
    {
        "category": "人际关系",
        "question": "我最好的朋友最近突然不跟我说话了，我问她也不理我，我做错什么了吗？",
        "expectation": "体谅困惑与自责，不预设对错，鼓励温和沟通",
    },
    # 【自我怀疑】
    {
        "category": "自我怀疑",
        "question": "我已经连续三次面试被拒了，是不是我真的很差劲？",
        "expectation": "接纳沮丧，区分\"面试失败\"和\"个人价值\"，引导复盘而非自我否定",
    },
    {
        "category": "自我怀疑",
        "question": "别人做什么都比我快比我好，我是不是天生就比别人笨？",
        "expectation": "共情比较带来的痛苦，打破\"天生笨\"的绝对化思维",
    },
    {
        "category": "自我怀疑",
        "question": "我总是害怕别人不喜欢我，所以说话做事都小心翼翼，好累。",
        "expectation": "理解讨好型人格的疲惫，引导关注自身需求",
    },
    # 【孤独/空虚】
    {
        "category": "孤独/空虚",
        "question": "室友都回家了，宿舍只剩我一个人，突然觉得特别孤单。",
        "expectation": "陪伴感优先，先共情孤独，不急于建议",
    },
    {
        "category": "孤独/空虚",
        "question": "周末一个人待在出租屋里，刷手机刷到想吐也不知道该做什么。",
        "expectation": "理解空虚感，引导从小行动开始，不指责\"浪费时间\"",
    },
    # 【情绪崩溃】
    {
        "category": "情绪崩溃",
        "question": "我觉得活着好累啊，每天睁开眼就觉得压力山大。",
        "expectation": "认真倾听不轻视，表达关心，引导寻求支持",
    },
    {
        "category": "情绪崩溃",
        "question": "我刚刚在电话里跟我妈吵了一架，现在在阳台上一个人哭。",
        "expectation": "先接纳情绪，不评判任何一方，鼓励冷静后再沟通",
    },
    {
        "category": "情绪崩溃",
        "question": "我什么都不想做，连起床刷牙都觉得累，已经三天了。",
        "expectation": "不催\"振作起来\"，共情疲惫，引导从小事开始",
    },
    # 【具体困扰】
    {
        "category": "具体困扰",
        "question": "我明天要在公司做 PPT 汇报，紧张到手心出汗，脑子一片空白。",
        "expectation": "理解紧张，给出临场减压小技巧和可操作的汇报准备步骤",
    },
    {
        "category": "具体困扰",
        "question": "医生说我需要做手术，但我特别害怕，不知道该跟谁说。",
        "expectation": "接纳恐惧，表达支持，鼓励与信任的人沟通、了解手术信息",
    },
    {
        "category": "具体困扰",
        "question": "我爸爸和妈妈又吵架了，我夹在中间不知道该帮谁，好难受。",
        "expectation": "体谅两难，不逼选边，引导保护自己的情绪边界",
    },
    # 【迷茫期】
    {
        "category": "迷茫期",
        "question": "毕业快一年了，换了三份工作，还是不知道自己想做什么。",
        "expectation": "不否定频繁换工作，引导从\"不喜欢什么\"倒推\"可能喜欢什么\"",
    },
    {
        "category": "迷茫期",
        "question": "我妈让我回老家考公务员，我不想回去但又怕自己在外面混不下去。",
        "expectation": "共情两难，不替做决定，引导梳理两种选择的代价和期待",
    },
    {
        "category": "迷茫期",
        "question": "我已经 25 岁了还没谈恋爱，是不是注定要单身一辈子？",
        "expectation": "接纳焦虑，打破\"25岁必须怎样\"的刻板认知，引导认识自己",
    },
]


async def call_model(url: str, api_key: str, prompt: str, model: str = "") -> str:
    """调用模型 API，返回回答文本。"""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    if model:
        payload["model"] = model

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                return f"[ERROR HTTP {resp.status_code}] {resp.text[:200]}"
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {str(e)[:200]}"


async def judge_response(question: str, expectation: str, baseline_answer: str,
                         finetuned_answer: str) -> dict:
    """用 DeepSeek 裁判对比打分。"""
    judge_prompt = f"""你是一名心理陪伴 AI 评测专家。请对比以下两个 AI 对用户问题的回答，从 3 个维度分别打分（1-5 分）。

评测维度说明：
- 共情性(Empathy)：是否先接纳情绪/感受，不急于说教或给建议。1=完全忽略情绪，5=深度共情并验证感受
- 建议合理性(Advice)：给出的建议是否具体、可执行、符合场景。1=无建议或建议空泛，5=具体可操作且贴合语境
- 人设一致性(Persona)：是否符合"温暖、真诚、非评判性的心理陪伴者"人设。1=冰冷说教/评判用户，5=温暖支持

用户问题：{question}
期望方向：{expectation}

--- 原版回答 ---
{baseline_answer[:500]}

--- 微调后回答 ---
{finetuned_answer[:500]}

请严格打分，对每个回答在每个维度给 1-5 分。输出 JSON 格式：
{{"baseline": {{"empathy": N, "advice": N, "persona": N, "notes": "一句话评价"}},
  "finetuned": {{"empathy": N, "advice": N, "persona": N, "notes": "一句话评价"}}}}
"""

    resp = await call_model(
        JUDGE_API_URL, JUDGE_API_KEY, judge_prompt, model="deepseek-chat"
    )

    # 解析 JSON（裁判可能输出 markdown 包裹的 JSON）
    resp = resp.strip()
    if resp.startswith("```"):
        resp = resp.strip("`")
        if resp.startswith("json"):
            resp = resp[4:].strip()

    try:
        data = json.loads(resp)
        return data
    except json.JSONDecodeError:
        # 尝试提取 JSON 对象
        import re
        match = re.search(r"\{[\s\S]*\}", resp)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"parse_error": resp[:300]}


async def main():
    print("=" * 60)
    print("微调评测：原版 Qwen2.5-7B-Instruct vs 微调后模型")
    print("=" * 60)

    # 1. 检查裁判 API
    if not JUDGE_API_KEY:
        print("[!] 未找到 DeepSeek API Key，请在 .env 中配置 LLM_API_KEY")
        sys.exit(1)
    print("[✓] 裁判 API: DeepSeek")

    # 2. 检查模型端点（只做格式校验，不测试连通性）
    print(f"[*] 原版模型: {BASELINE_API_URL}")
    print(f"[*] 微调后模型: {FINETUNED_API_URL}")
    print("\n请先部署两个模型为 OpenAI 兼容 API，然后修改脚本中的 URL 配置再运行。")
    print("部署命令示例（vLLM）：")
    print("  vllm serve --served-model-name qwen2.5-base <原版模型路径> --port 8080")
    print("  vllm serve --served-model-name qwen2.5-ft <微调模型路径> --port 8081")
    print()

    # 3. 逐个评测
    print(f"开始评测（共 {len(EVAL_QUESTIONS)} 题）...")
    print("-" * 60)

    results = []
    dims = ["empathy", "advice", "persona"]

    for i, case in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] {case['category']}")
        print(f"  Q: {case['question'][:60]}...")

        # 获取两个回答
        baseline = await call_model(BASELINE_API_URL, BASELINE_API_KEY, case["question"])
        finetuned = await call_model(FINETUNED_API_URL, FINETUNED_API_KEY, case["question"])

        if baseline.startswith("[ERROR") or finetuned.startswith("[ERROR"):
            print(f"  [!] API 调用失败，跳过此题")
            results.append({
                "index": i,
                "category": case["category"],
                "question": case["question"],
                "baseline_answer": baseline,
                "finetuned_answer": finetuned,
                "error": True,
            })
            continue

        # 裁判打分
        judge = await judge_response(
            case["question"], case["expectation"], baseline, finetuned
        )

        b = judge.get("baseline", {})
        f = judge.get("finetuned", {})

        b_avg = sum(b.get(d, 0) for d in dims) / 3
        f_avg = sum(f.get(d, 0) for d in dims) / 3

        print(f"  原版: 共情{b.get('empathy','?')} 建议{b.get('advice','?')} 人设{b.get('persona','?')} | 均值{b_avg:.1f}")
        print(f"  微调: 共情{f.get('empathy','?')} 建议{f.get('advice','?')} 人设{f.get('persona','?')} | 均值{f_avg:.1f}")
        if f_avg > b_avg:
            print(f"  → 微调 +{f_avg - b_avg:.1f} ✓")
        else:
            print(f"  → 微调 {f_avg - b_avg:.1f} ✗")

        results.append({
            "index": i,
            "category": case["category"],
            "question": case["question"],
            "expectation": case["expectation"],
            "baseline_answer": baseline,
            "finetuned_answer": finetuned,
            "baseline_scores": b,
            "finetuned_scores": f,
            "baseline_avg": round(b_avg, 2),
            "finetuned_avg": round(f_avg, 2),
        })

    # 4. 汇总统计
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print("\n[!] 无有效评测结果")
        return

    def avg(rows, key, dim):
        vals = [r[key].get(dim, 0) for r in rows if key in r and isinstance(r[key], dict)]
        return sum(vals) / len(vals) if vals else 0

    print("\n" + "=" * 60)
    print("评测结果汇总")
    print("=" * 60)

    print(f"""
┌──────────────────┬───────────┬───────────┬───────────┬───────────┐
│ 维度             │ 原版均分   │ 微调均分   │ 提升      │ 提升率    │
├──────────────────┼───────────┼───────────┼───────────┼───────────┤
│ 共情性 Empathy   │ {avg(valid, 'baseline_scores', 'empathy'):.1f}       │ {avg(valid, 'finetuned_scores', 'empathy'):.1f}       │ {avg(valid, 'finetuned_scores', 'empathy') - avg(valid, 'baseline_scores', 'empathy'):+.1f}      │ │
│ 建议合理性 Advice │ {avg(valid, 'baseline_scores', 'advice'):.1f}       │ {avg(valid, 'finetuned_scores', 'advice'):.1f}       │ {avg(valid, 'finetuned_scores', 'advice') - avg(valid, 'baseline_scores', 'advice'):+.1f}      │ │
│ 人设一致性 Persona│ {avg(valid, 'baseline_scores', 'persona'):.1f}       │ {avg(valid, 'finetuned_scores', 'persona'):.1f}       │ {avg(valid, 'finetuned_scores', 'persona') - avg(valid, 'baseline_scores', 'persona'):+.1f}      │ │
├──────────────────┼───────────┼───────────┼───────────┼───────────┤
│ 综合均分          │ {sum(r['baseline_avg'] for r in valid)/len(valid):.1f}       │ {sum(r['finetuned_avg'] for r in valid)/len(valid):.1f}       │ {sum(r['finetuned_avg'] - r['baseline_avg'] for r in valid)/len(valid):+.1f}      │ │
└──────────────────┴───────────┴───────────┴───────────┴───────────┘

评测环境: DeepSeek 裁判评分 (1-5 分)
评测用例: {len(valid)} 道心理陪伴场景问题
模型: Qwen2.5-7B-Instruct (原版 vs LoRA 微调合并版)
训练数据: 灵心心理健康多轮对话数据集
训练配置: LoRA rank=8, 42 线性层, 2 轮, bf16, lr=2e-5
损失变化: 2.65 → 1.33
""")

    # 按类别统计
    from collections import defaultdict
    cat_stats = defaultdict(list)
    for r in valid:
        cat_stats[r["category"]].append((r["baseline_avg"], r["finetuned_avg"]))

    print("--- 按场景类别 ---")
    for cat, scores in cat_stats.items():
        b_avg = sum(s[0] for s in scores) / len(scores)
        f_avg = sum(s[1] for s in scores) / len(scores)
        print(f"  {cat}: {b_avg:.1f} → {f_avg:.1f} ({len(scores)}题)")

    # 5. 保存结果
    output_path = PROJECT_ROOT / "data" / "output" / "eval_finetune_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "baseline_api": BASELINE_API_URL,
                "finetuned_api": FINETUNED_API_URL,
                "judge": "deepseek-chat",
                "n_questions": len(EVAL_QUESTIONS),
            },
            "dim_averages": {
                "baseline_empathy": avg(valid, "baseline_scores", "empathy"),
                "finetuned_empathy": avg(valid, "finetuned_scores", "empathy"),
                "baseline_advice": avg(valid, "baseline_scores", "advice"),
                "finetuned_advice": avg(valid, "finetuned_scores", "advice"),
                "baseline_persona": avg(valid, "baseline_scores", "persona"),
                "finetuned_persona": avg(valid, "finetuned_scores", "persona"),
            },
            "overall": {
                "baseline": round(sum(r["baseline_avg"] for r in valid) / len(valid), 2),
                "finetuned": round(sum(r["finetuned_avg"] for r in valid) / len(valid), 2),
                "delta": round(sum(r["finetuned_avg"] - r["baseline_avg"] for r in valid) / len(valid), 2),
            },
            "category_stats": {
                cat: {
                    "baseline": round(sum(s[0] for s in scores) / len(scores), 2),
                    "finetuned": round(sum(s[1] for s in scores) / len(scores), 2),
                }
                for cat, scores in cat_stats.items()
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
