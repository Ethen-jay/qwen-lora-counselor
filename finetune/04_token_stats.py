import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

# 尝试导入 transformers 的 tokenizer
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def clean_text(text: str) -> str:
    """与 Clean.py 中相同的文本清理函数（这里主要用于统一换行等，不影响长度统计）"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    return text

def compute_token_length(sample: Dict, tokenizer=None) -> int:
    """
    计算一个样本的总 token 数。
    如果提供了 tokenizer，使用 tokenizer 编码；否则退化为字符数估算。
    """
    conversations = sample.get('conversations', [])
    if not conversations:
        return 0

    all_text = ""
    for msg in conversations:
        value = msg.get('value', '')
        all_text += value + "\n"

    if tokenizer is not None:
        try:
            return len(tokenizer.encode(all_text))
        except:
            pass
    # 退化为字符数（粗略估算）
    return len(all_text)

def load_tokenizer(tokenizer_name: Optional[str]):
    """
    尝试加载 tokenizer。如果未指定或加载失败，返回 None。
    """
    if not tokenizer_name:
        return None
    if not TRANSFORMERS_AVAILABLE:
        print("警告：transformers 库未安装，将使用字符数估算 token 长度。")
        return None
    try:
        print(f"正在加载 tokenizer：{tokenizer_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return tokenizer
    except Exception as e:
        print(f"加载 tokenizer 失败：{e}")
        print("将使用字符数估算 token 长度。")
        return None

def print_text_histogram(lengths: List[int], bins=20):
    """
    打印 ASCII 直方图，展示 token 长度分布。
    """
    if not lengths:
        return
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len == max_len:
        print(f"所有样本 token 长度均为 {min_len}")
        return

    bin_width = max(1, (max_len - min_len) // bins)
    bin_edges = list(range(min_len, max_len + bin_width, bin_width))
    if len(bin_edges) <= 1:
        bin_edges = [min_len, max_len + 1]
    bin_counts = [0] * (len(bin_edges) - 1)
    for l in lengths:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= l < bin_edges[i+1]:
                bin_counts[i] += 1
                break
        else:
            # 最大值情况
            bin_counts[-1] += 1

    max_count = max(bin_counts) if bin_counts else 1
    print("\nToken 长度分布直方图（每个 * 代表 {} 个样本）：".format(max(1, max_count // 50)))
    for i in range(len(bin_counts)):
        left = bin_edges[i]
        right = bin_edges[i+1] - 1
        bar = '*' * max(1, int(bin_counts[i] / max(1, max_count / 50)))
        print(f"{left:>6}-{right:<6} | {bar} ({bin_counts[i]})")

def main():
    parser = argparse.ArgumentParser(description='统计 ShareGPT 格式对话数据集的 token 长度分布')
    parser.add_argument('--input', type=str, required=True,
                        help='输入 JSONL 文件路径')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='HuggingFace tokenizer 名称或路径（如 Qwen/Qwen2.5-7B-Instruct），用于精确计算 token 长度')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='可选：统计时报告超过该阈值样本的数量和比例')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式（不打印直方图，只输出统计数字）')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在：{args.input}")
        return

    tokenizer = load_tokenizer(args.tokenizer)

    print(f"[读取] 开始读取 {args.input} ...")
    total_count = 0
    invalid_rows = 0
    lengths = []

    with open(args.input, 'r', encoding='utf-8') as fin:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total_count += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue

            # 计算 token 长度（即使格式不完整也计算，但 conversations 缺失则长度为 0）
            token_len = compute_token_length(sample, tokenizer)
            lengths.append(token_len)

    if invalid_rows > 0:
        print(f"警告：{invalid_rows} 行 JSON 解析失败，已跳过。")

    if not lengths:
        print("没有有效样本。")
        return

    # 统计数据
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    total_tokens = sum(lengths_sorted)
    avg = total_tokens / n
    min_len = lengths_sorted[0]
    max_len = lengths_sorted[-1]

    def percentile(p):
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f
        if f == c:
            return lengths_sorted[f]
        return lengths_sorted[f] + (lengths_sorted[c] - lengths_sorted[f]) * (k - f)

    median = percentile(50)
    p25 = percentile(25)
    p75 = percentile(75)
    p90 = percentile(90)
    p95 = percentile(95)
    p99 = percentile(99)

    print("\n" + "="*50)
    print(f"Token 长度统计结果")
    print(f"样本总数：{n}")
    print(f"平均 token 数：{avg:.2f}")
    print(f"最小 token 数：{min_len}")
    print(f"最大 token 数：{max_len}")
    print(f"中位数（50%）：{median:.2f}")
    print(f"25% 分位数：{p25:.2f}")
    print(f"75% 分位数：{p75:.2f}")
    print(f"90% 分位数：{p90:.2f}")
    print(f"95% 分位数：{p95:.2f}")
    print(f"99% 分位数：{p99:.2f}")

    if args.max_tokens is not None:
        over = sum(1 for l in lengths if l > args.max_tokens)
        print(f"\n超过 {args.max_tokens} token 的样本数：{over} ({over/n*100:.2f}%)")

    if not args.quiet:
        print_text_histogram(lengths, bins=20)

    print("\n统计完成。")

if __name__ == '__main__':
    main()