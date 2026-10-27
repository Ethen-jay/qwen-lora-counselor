import json
import argparse
import os

def convert_sharegpt_to_alpaca(input_file, output_file, keep_all_turns=False):
    """
    将 ShareGPT 格式的 JSONL 转换为 Alpaca 格式。
    - input_file: 输入的 ShareGPT JSONL 文件路径
    - output_file: 输出的 Alpaca JSONL 文件路径
    - keep_all_turns: 是否将多轮对话合并为一个 output（不推荐，默认 False）
        False: 仅保留第一轮 human 和第一轮 gpt
        True: 将所有 human 消息合并为 instruction，所有 gpt 消息合并为 output
    """
    total = 0
    skipped = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[警告] 第 {line_num} 行 JSON 解析失败，已跳过。错误：{e}")
                skipped += 1
                continue

            conversations = data.get('conversations')
            if not isinstance(conversations, list):
                print(f"[警告] 第 {line_num} 行缺少有效的 'conversations' 字段，已跳过。")
                skipped += 1
                continue

            # 提取所有 human 和 gpt 消息
            human_msgs = []
            gpt_msgs = []
            for msg in conversations:
                if not isinstance(msg, dict):
                    continue
                role = msg.get('from')
                value = msg.get('value', '')
                if role == 'human':
                    human_msgs.append(value)
                elif role == 'gpt':
                    gpt_msgs.append(value)
                # 忽略 system, observation 等其他角色

            if not human_msgs or not gpt_msgs:
                print(f"[警告] 第 {line_num} 行缺少 human 或 gpt 消息，已跳过。")
                skipped += 1
                continue

            if keep_all_turns:
                # 合并所有轮次（不推荐）
                instruction = '\n'.join(human_msgs)
                output = '\n'.join(gpt_msgs)
            else:
                # 仅保留第一轮
                instruction = human_msgs[0]
                output = gpt_msgs[0]

            alpaca_item = {
                "instruction": instruction,
                "input": "",
                "output": output
            }

            fout.write(json.dumps(alpaca_item, ensure_ascii=False) + '\n')
            total += 1

    print(f"转换完成！")
    print(f"成功转换：{total} 条样本")
    print(f"跳过样本：{skipped} 条（原因：格式错误或缺少必需字段）")
    print(f"输出文件：{output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ShareGPT 数据转 Alpaca 格式')
    parser.add_argument('--input', type=str, required=True, help='输入 ShareGPT JSONL 文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出 Alpaca JSONL 文件路径')
    parser.add_argument('--keep_all_turns', action='store_true', 
                        help='是否合并所有轮次（默认只取第一轮）')
    args = parser.parse_args()

    convert_sharegpt_to_alpaca(args.input, args.output, args.keep_all_turns)