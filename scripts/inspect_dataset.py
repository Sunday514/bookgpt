from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from bookgpt.common import count_text_tokens, load_tokenizer, read_jsonl


def count_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(str(message.get('content', ''))) for message in messages)


def count_turns(messages: list[dict[str, str]]) -> int:
    return len(messages)


def validate_record(record: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    messages = record.get('messages')
    if not isinstance(messages, list) or not messages:
        issues.append('missing_messages')
        return issues

    last_role = None
    for idx, message in enumerate(messages):
        role = message.get('role')
        content = str(message.get('content', '')).strip()
        if role not in {'system', 'user', 'assistant'}:
            issues.append('bad_role')
        if not content:
            issues.append('empty_content')
        if last_role == role:
            issues.append('duplicate_adjacent_role')
        last_role = role
        if idx == len(messages) - 1 and role != 'assistant':
            issues.append('last_not_assistant')

    if 'text' not in record or not str(record['text']).strip():
        issues.append('missing_text')

    meta = record.get('meta')
    if not isinstance(meta, dict):
        issues.append('missing_meta')
        return issues

    if not meta.get('source'):
        issues.append('missing_source')
    if not meta.get('task_type'):
        issues.append('missing_task_type')

    holdout = meta.get('safety', {}).get('memorization_holdout', False)
    if holdout and meta.get('split') == 'train':
        issues.append('holdout_in_train')

    return sorted(set(issues))


def inspect_split(name: str, records: list[dict[str, Any]], tokenizer=None) -> dict[str, Any]:
    source_counter = Counter()
    task_counter = Counter()
    role_pattern_counter = Counter()
    issue_counter = Counter()
    char_counts: list[int] = []
    turn_counts: list[int] = []
    token_counts: list[int] = []

    for record in records:
        meta = record.get('meta', {})
        source_counter[meta.get('source', 'unknown')] += 1
        task_counter[meta.get('task_type', 'unknown')] += 1
        role_pattern = '>'.join(message.get('role', '?') for message in record.get('messages', []))
        role_pattern_counter[role_pattern] += 1
        char_counts.append(count_chars(record.get('messages', [])))
        turn_counts.append(count_turns(record.get('messages', [])))
        if tokenizer is not None:
            token_counts.append(count_text_tokens(str(record.get('text', '')), tokenizer))
        elif meta.get('token_count'):
            token_counts.append(int(meta['token_count']))
        for issue in validate_record(record):
            issue_counter[issue] += 1

    summary = {
        'records': len(records),
        'sources': dict(source_counter),
        'tasks': dict(task_counter),
        'top_role_patterns': dict(role_pattern_counter.most_common(5)),
        'issues': dict(issue_counter),
    }
    if char_counts:
        summary['message_char_stats'] = {
            'min': min(char_counts),
            'median': median(char_counts),
            'mean': round(mean(char_counts), 2),
            'max': max(char_counts),
        }
    if turn_counts:
        summary['turn_stats'] = {
            'min': min(turn_counts),
            'median': median(turn_counts),
            'mean': round(mean(turn_counts), 2),
            'max': max(turn_counts),
        }
    if token_counts:
        summary['token_stats'] = {
            'min': min(token_counts),
            'median': median(token_counts),
            'mean': round(mean(token_counts), 2),
            'max': max(token_counts),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect a Qwen SFT JSONL dataset directory.')
    parser.add_argument('--dataset-dir', required=True, help='Directory containing train/dev/test JSONL files.')
    parser.add_argument('--tokenizer-path', help='Tokenizer path used to compute token statistics.')
    parser.add_argument('--write-summary', action='store_true', help='Write dataset_summary.json to the dataset directory.')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    tokenizer = load_tokenizer(args.tokenizer_path) if args.tokenizer_path else None
    splits = {
        'train': read_jsonl(dataset_dir / 'train.jsonl'),
        'dev': read_jsonl(dataset_dir / 'dev.jsonl'),
        'test': read_jsonl(dataset_dir / 'test.jsonl'),
    }
    summary = {
        'dataset_dir': str(dataset_dir),
        'tokenizer_path': args.tokenizer_path,
        'splits': {name: inspect_split(name, records, tokenizer=tokenizer) for name, records in splits.items()},
    }

    if args.write_summary:
        target = dataset_dir / 'dataset_summary.json'
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
