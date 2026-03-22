from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import read_jsonl, write_jsonl


def build_formatted_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message['role']
        content = message['content'].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return '\n'.join(parts)


def split_records(records: list[dict], train_ratio: float, dev_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    total = len(records)
    if total < 3:
        raise ValueError('Need at least 3 samples to produce train/dev/test splits.')

    train_end = int(total * train_ratio)
    dev_size = int(total * dev_ratio)

    train_end = min(max(train_end, 1), total - 2)
    dev_size = min(max(dev_size, 1), total - train_end - 1)
    dev_end = train_end + dev_size

    train = records[:train_end]
    dev = records[train_end:dev_end]
    test = records[dev_end:]
    return train, dev, test


def main() -> None:
    parser = argparse.ArgumentParser(description='Build starter train/dev/test JSONL files.')
    parser.add_argument('--samples', default='dataset/samples.jsonl')
    parser.add_argument('--output-dir', default='dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--dev-ratio', type=float, default=0.1)
    args = parser.parse_args()

    records = read_jsonl(args.samples)
    for record in records:
        record['text'] = build_formatted_text(record['messages'])

    train, dev, test = split_records(records, args.train_ratio, args.dev_ratio)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / 'train.jsonl', train)
    write_jsonl(output_dir / 'dev.jsonl', dev)
    write_jsonl(output_dir / 'test.jsonl', test)

    split_counter = Counter(record['meta'].get('task_type', 'unknown') for record in records)
    print(f'Wrote {len(train)} train / {len(dev)} dev / {len(test)} test records to {output_dir}')
    print('Task type distribution:', dict(split_counter))


if __name__ == '__main__':
    main()
