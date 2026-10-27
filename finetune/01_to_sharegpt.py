import argparse
import json
import os

# 数据集文件路径配置（请根据实际文件位置调整）
DATASET_FILES = {
    'psydt': 'PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json',
    # 可添加其他数据集
}

def to_sharegpt(sample):
    """
    将原始多轮 messages 格式转换为 ShareGPT 格式。
    输入样例：
        {
            "id": 0,
            "normalizedTag": "婚恋",
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }
    输出：
        {
            "conversations": [
                {"from": "system", "value": "..."},
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."},
                ...
            ]
        }
    """
    if not isinstance(sample, dict):
        return None

    messages = sample.get('messages')
    if not messages or not isinstance(messages, list):
        return None

    conversations = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role', '').lower()
        content = msg.get('content', '').strip()
        if not content:
            continue

        # ShareGPT 角色映射
        if role == 'user':
            from_role = 'human'
        elif role == 'assistant':
            from_role = 'gpt'
        elif role == 'system':
            from_role = 'system'
        else:
            continue  # 忽略未知角色

        conversations.append({
            'from': from_role,
            'value': content
        })

    if conversations:
        return {'conversations': conversations}
    return None


def main():
    parser = argparse.ArgumentParser(description='转换数据集为 ShareGPT 格式（llamaFactory 支持）')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称，如 psydt')
    parser.add_argument('--output', type=str, default='sharegpt_converted.jsonl',
                        help='输出文件路径（默认 sharegpt_converted.jsonl）')
    args = parser.parse_args()

    if args.dataset not in DATASET_FILES:
        print(f"错误：未知数据集 '{args.dataset}'")
        print(f"可用数据集：{list(DATASET_FILES.keys())}")
        return

    raw_file = DATASET_FILES[args.dataset]
    if not os.path.exists(raw_file):
        print(f"错误：原始文件不存在：{raw_file}")
        return

    # 读取原始 JSON 文件（应为列表形式）
    print(f"[读取] 正在读取 {raw_file}...")
    with open(raw_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("错误：原始 JSON 文件的顶层结构不是列表")
        return

    total = len(data)
    print(f"[读取] 原始样本数：{total}")

    # 转换
    converted = []
    for sample in data:
        result = to_sharegpt(sample)
        if result:
            converted.append(result)

    success = len(converted)
    print(f"[转换] 成功转换 {success} 条样本（失败 {total - success} 条）")

    # 写入 JSONL 文件
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"[完成] ShareGPT 格式数据已保存至：{args.output}")


if __name__ == '__main__':
    main()