import argparse
import json
import os
import re
import hashlib
from typing import Dict, List, Optional, Tuple

# 尝试导入 transformers 的 tokenizer，若未安装则退化为简单字符长度估算
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def clean_text(text: str) -> str:
    """
    清理文本中的控制字符、多余空格、统一换行符。
    """
    if not isinstance(text, str):
        return ""
    # 去除控制字符（保留换行符和制表符）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 将多个空格或制表符压缩为一个空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 去除空行（保留正常段落）
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

def is_valid_sample(sample: Dict) -> bool:
    """
    检查样本是否至少包含一轮 human 和 gpt 的对话。
    """
    conversations = sample.get('conversations', [])
    if not isinstance(conversations, list) or len(conversations) < 2:
        return False

    has_human = False
    has_gpt = False
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        role = msg.get('from', '').lower()
        if role == 'human':
            has_human = True
        elif role == 'gpt':
            has_gpt = True
        if has_human and has_gpt:
            return True
    return False

def hash_sample(sample: Dict) -> str:
    """
    生成样本内容的哈希值，用于去重。
    """
    conversations = sample.get('conversations', [])
    content = ""
    for msg in conversations:
        if isinstance(msg, dict):
            content += f"{msg.get('from','')}:{msg.get('value','')}\n"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

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

def main():
    parser = argparse.ArgumentParser(description='清洗 ShareGPT 格式的对话数据集')
    parser.add_argument('--input', type=str, required=True,
                        help='输入 JSONL 文件路径')
    parser.add_argument('--output', type=str, default='cleaned_data.jsonl',
                        help='输出 JSONL 文件路径（默认：cleaned_data.jsonl）')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='最大 token 数（超过则丢弃样本），不设置则不限制')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='HuggingFace tokenizer 名称或路径（如 Qwen/Qwen2.5-7B-Instruct），用于精确计算 token 长度')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式（不打印每个过滤样本的详细信息）')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在：{args.input}")
        return

    # 加载 tokenizer（可选）
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"[读取] 开始读取 {args.input} ...")
    total_count = 0
    clean_count = 0
    removed_invalid = 0
    removed_too_long = 0
    removed_duplicate = 0
    seen_hashes = set()

    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total_count += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析失败，已跳过：{e}")
                removed_invalid += 1
                continue

            # 1. 基本格式校验
            if not is_valid_sample(sample):
                if not args.quiet:
                    print(f"第 {line_num} 行：缺少有效的 human/gpt 对话，已过滤")
                removed_invalid += 1
                continue

            # 2. 清理文本内容
            conversations = sample.get('conversations', [])
            for msg in conversations:
                if isinstance(msg, dict) and 'value' in msg:
                    msg['value'] = clean_text(msg['value'])

            # 3. 长度过滤
            if args.max_tokens is not None:
                token_len = compute_token_length(sample, tokenizer)
                if token_len > args.max_tokens:
                    if not args.quiet:
                        print(f"第 {line_num} 行：token 长度 {token_len} 超过阈值 {args.max_tokens}，已过滤")
                    removed_too_long += 1
                    continue

            # 4. 去重
            sample_hash = hash_sample(sample)
            if sample_hash in seen_hashes:
                if not args.quiet:
                    print(f"第 {line_num} 行：重复样本，已过滤")
                removed_duplicate += 1
                continue
            seen_hashes.add(sample_hash)

            # 输出干净样本
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
            clean_count += 1

    print("\n" + "="*50)
    print(f"清洗完成！")
    print(f"原始样本总数：{total_count}")
    print(f"保留样本数：{clean_count}")
    print(f"过滤原因统计：")
    print(f"  - 无效格式/缺少对话：{removed_invalid}")
    print(f"  - 超过最大 token 数：{removed_too_long}")
    print(f"  - 重复样本：{removed_duplicate}")
    print(f"输出文件：{args.output}")

if __name__ == '__main__':
    main()