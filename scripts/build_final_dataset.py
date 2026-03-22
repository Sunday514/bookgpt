from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from common import annotate_token_counts, load_tokenizer, read_jsonl, write_jsonl

DEFAULT_BOOK_DIR = Path('dataset/book_run')
DEFAULT_EXTERNAL_DIR = Path('dataset/external_normalized')

OPENHERMES_FUNCTION_RE = re.compile(r'<functioncall>|FUNCTION RESPONSE|外部函数|tool call|function calling', re.I)
OPENHERMES_CODE_RE = re.compile(r'```|\bpython\b|\bjavascript\b|\bjava\b|\blatex\b|\bsql\b|代码|编程', re.I)
OPENHERMES_WRITING_RE = re.compile(r'改写|扩写|润色|写一篇|写一段|诗|小说|提纲|续写|文案|摘要|故事|剧本|散文')


def validate_final_records(split_name: str, records: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for index, record in enumerate(records):
        messages = record.get('messages', [])
        if not messages:
            issues.append(f'{split_name}[{index}]:missing_messages')
            continue
        if messages[-1].get('role') != 'assistant':
            issues.append(f'{split_name}[{index}]:last_not_assistant')
        if not record.get('text'):
            issues.append(f'{split_name}[{index}]:missing_text')
        meta = record.get('meta', {})
        holdout = meta.get('safety', {}).get('memorization_holdout', False)
        if split_name == 'train' and holdout:
            issues.append(f'{split_name}[{index}]:holdout_in_train')
    return issues


def split_external_records(records: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total == 0:
        return [], [], []
    if total == 1:
        return shuffled, [], []
    if total == 2:
        return [shuffled[0]], [shuffled[1]], []

    dev_count = max(1, round(total * 0.05))
    test_count = max(1, round(total * 0.05))
    if dev_count + test_count >= total:
        test_count = 1
        dev_count = 1
    train_count = total - dev_count - test_count
    return shuffled[:train_count], shuffled[train_count:train_count + dev_count], shuffled[train_count + dev_count:]


def classify_openhermes(record: dict[str, Any]) -> str:
    text = '\n'.join(message['content'] for message in record.get('messages', []))
    if OPENHERMES_FUNCTION_RE.search(text):
        return 'function'
    if OPENHERMES_CODE_RE.search(text):
        return 'code'
    if OPENHERMES_WRITING_RE.search(text):
        return 'writing'
    return 'general'


def filter_openhermes(records: list[dict[str, Any]], allow_code: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    stats = Counter()
    for record in records:
        category = classify_openhermes(record)
        stats[f'seen_{category}'] += 1
        if category == 'function':
            stats['dropped_function'] += 1
            continue
        if category == 'code' and not allow_code:
            stats['dropped_code'] += 1
            continue
        updated = dict(record)
        meta = dict(updated.get('meta', {}))
        meta['openhermes_category'] = category
        updated['meta'] = meta
        kept.append(updated)
        stats[f'kept_{category}'] += 1
    stats['kept_total'] = len(kept)
    return kept, dict(stats)


def filter_by_token_length(
    records: list[dict[str, Any]],
    tokenizer,
    *,
    max_tokens: int | None,
    min_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tokenizer is None:
        return records, {
            'enabled': False,
            'kept': len(records),
            'dropped_too_short': 0,
            'dropped_too_long': 0,
        }

    annotated = annotate_token_counts(records, tokenizer)
    kept: list[dict[str, Any]] = []
    token_counts: list[int] = []
    dropped_too_short = 0
    dropped_too_long = 0

    for record in annotated:
        token_count = int(record.get('meta', {}).get('token_count', 0))
        if token_count < min_tokens:
            dropped_too_short += 1
            continue
        if max_tokens is not None and token_count > max_tokens:
            dropped_too_long += 1
            continue
        kept.append(record)
        token_counts.append(token_count)

    stats: dict[str, Any] = {
        'enabled': True,
        'kept': len(kept),
        'dropped_too_short': dropped_too_short,
        'dropped_too_long': dropped_too_long,
        'min_tokens': min_tokens,
        'max_tokens': max_tokens,
    }
    if token_counts:
        stats['token_stats'] = {
            'min': min(token_counts),
            'median': median(token_counts),
            'mean': round(mean(token_counts), 2),
            'max': max(token_counts),
        }
    return kept, stats


def sample_records(records: list[dict[str, Any]], target: int, seed: int, allow_repeat: bool) -> list[dict[str, Any]]:
    if target <= 0 or not records:
        return []
    rng = random.Random(seed)
    if allow_repeat and len(records) < target:
        return [rng.choice(records) for _ in range(target)]
    if len(records) <= target:
        return list(records)
    return rng.sample(records, target)


def parse_ratios(value: str) -> dict[str, float]:
    parts = [item.strip() for item in value.split(',') if item.strip()]
    if len(parts) != 4:
        raise ValueError('ratios must contain exactly 4 comma-separated numbers: book,openhermes,coig_writer,oasst1_zh')
    keys = ['book', 'openhermes', 'coig_writer', 'oasst1_zh']
    ratios = {key: float(part) for key, part in zip(keys, parts)}
    if any(v < 0 for v in ratios.values()):
        raise ValueError('ratios must be non-negative')
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError('ratios must sum to a positive value')
    return {key: item / total for key, item in ratios.items()}


def compute_targets(target_total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {key: target_total * ratio for key, ratio in ratios.items()}
    floors = {key: math.floor(value) for key, value in raw.items()}
    remainder = target_total - sum(floors.values())
    order = sorted(raw, key=lambda key: (raw[key] - floors[key]), reverse=True)
    for key in order[:remainder]:
        floors[key] += 1
    return floors


def allocate_eval_target(train_target: int) -> int:
    return max(0, round(train_target * 0.05))


def summarize_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_counter = Counter(record.get('meta', {}).get('source', 'unknown') for record in records)
    task_counter = Counter(record.get('meta', {}).get('task_type', 'unknown') for record in records)
    token_counts = [int(record.get('meta', {}).get('token_count', 0)) for record in records if record.get('meta', {}).get('token_count')]
    summary = {
        'records': len(records),
        'sources': dict(source_counter),
        'tasks': dict(task_counter),
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
    parser = argparse.ArgumentParser(description='Build the final Qwen training dataset from book and external sources.')
    parser.add_argument('--book-dir', default=str(DEFAULT_BOOK_DIR), help='Directory with book train/dev/test JSONL files.')
    parser.add_argument('--external-dir', default=str(DEFAULT_EXTERNAL_DIR), help='Directory with normalized external JSONL files.')
    parser.add_argument('--output-dir', required=True, help='Directory for final train/dev/test JSONL files.')
    parser.add_argument('--target-total', type=int, default=10000, help='Target number of training records in the final train split.')
    parser.add_argument(
        '--ratios',
        default='0.45,0.20,0.25,0.10',
        help='Comma-separated train ratios for book,openhermes,coig_writer,oasst1_zh',
    )
    parser.add_argument('--allow-openhermes-code', action='store_true', help='Keep code-heavy OpenHermes samples.')
    parser.add_argument('--tokenizer-path', help='Tokenizer path used for token-length filtering and reporting.')
    parser.add_argument('--max-tokens', type=int, help='Drop samples whose text token count exceeds this value.')
    parser.add_argument('--min-tokens', type=int, default=1, help='Drop samples whose text token count is below this value.')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    if args.target_total <= 0:
        raise ValueError('target-total must be > 0')
    if args.min_tokens < 0:
        raise ValueError('min-tokens must be >= 0')
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError('max-tokens must be > 0')
    if args.max_tokens is not None and args.min_tokens > args.max_tokens:
        raise ValueError('min-tokens must be <= max-tokens')

    ratios = parse_ratios(args.ratios)

    book_dir = Path(args.book_dir)
    external_dir = Path(args.external_dir)
    output_dir = Path(args.output_dir)

    tokenizer = load_tokenizer(args.tokenizer_path) if args.tokenizer_path else None

    book_train_raw = read_jsonl(book_dir / 'train.jsonl')
    book_dev_raw = read_jsonl(book_dir / 'dev.jsonl')
    book_test_raw = read_jsonl(book_dir / 'test.jsonl')

    book_train, book_train_filter_stats = filter_by_token_length(
        book_train_raw,
        tokenizer,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    book_dev, book_dev_filter_stats = filter_by_token_length(
        book_dev_raw,
        tokenizer,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    book_test, book_test_filter_stats = filter_by_token_length(
        book_test_raw,
        tokenizer,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    book_fallback_mode = False
    if not book_train:
        book_fallback_mode = True

    openhermes_all = read_jsonl(external_dir / 'openhermes_2_5_zh.jsonl')
    oasst_all = read_jsonl(external_dir / 'oasst1_zh.jsonl')
    coig_all = read_jsonl(external_dir / 'coig_writer.jsonl')

    openhermes_filtered, openhermes_filter_stats = filter_openhermes(openhermes_all, args.allow_openhermes_code)
    openhermes_filtered, openhermes_token_filter_stats = filter_by_token_length(
        openhermes_filtered,
        tokenizer,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    oasst_all, oasst_token_filter_stats = filter_by_token_length(
        oasst_all,
        tokenizer,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    coig_all, coig_token_filter_stats = filter_by_token_length(
        coig_all,
        tokenizer,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )

    oh_train, oh_dev, oh_test = split_external_records(openhermes_filtered, args.seed)
    oa_train, oa_dev, oa_test = split_external_records(oasst_all, args.seed + 1)
    co_train, co_dev, co_test = split_external_records(coig_all, args.seed + 2)

    effective_ratios = dict(ratios)
    if book_fallback_mode:
        effective_ratios['book'] = 0.0
        remaining = effective_ratios['openhermes'] + effective_ratios['coig_writer'] + effective_ratios['oasst1_zh']
        if remaining <= 0:
            raise ValueError('No available data source remains after disabling the empty book split.')
        effective_ratios = {
            key: (value / remaining if key != 'book' else 0.0)
            for key, value in effective_ratios.items()
        }

    train_targets = compute_targets(args.target_total, effective_ratios)

    train_records = []
    train_records += sample_records(book_train, train_targets['book'], args.seed + 10, allow_repeat=True)
    train_records += sample_records(oh_train, train_targets['openhermes'], args.seed + 11, allow_repeat=False)
    train_records += sample_records(co_train, train_targets['coig_writer'], args.seed + 12, allow_repeat=True)
    train_records += sample_records(oa_train, train_targets['oasst1_zh'], args.seed + 13, allow_repeat=True)

    dev_targets = {
        'book': min(len(book_dev), allocate_eval_target(train_targets['book'])),
        'openhermes': min(len(oh_dev), allocate_eval_target(train_targets['openhermes'])),
        'coig_writer': min(len(co_dev), allocate_eval_target(train_targets['coig_writer'])),
        'oasst1_zh': min(len(oa_dev), allocate_eval_target(train_targets['oasst1_zh'])),
    }
    test_targets = {
        'book': min(len(book_test), allocate_eval_target(train_targets['book'])),
        'openhermes': min(len(oh_test), allocate_eval_target(train_targets['openhermes'])),
        'coig_writer': min(len(co_test), allocate_eval_target(train_targets['coig_writer'])),
        'oasst1_zh': min(len(oa_test), allocate_eval_target(train_targets['oasst1_zh'])),
    }

    dev_records = []
    dev_records += sample_records(book_dev, dev_targets['book'], args.seed + 20, allow_repeat=False)
    dev_records += sample_records(oh_dev, dev_targets['openhermes'], args.seed + 21, allow_repeat=False)
    dev_records += sample_records(co_dev, dev_targets['coig_writer'], args.seed + 22, allow_repeat=False)
    dev_records += sample_records(oa_dev, dev_targets['oasst1_zh'], args.seed + 23, allow_repeat=False)

    test_records = []
    test_records += sample_records(book_test, test_targets['book'], args.seed + 30, allow_repeat=False)
    test_records += sample_records(oh_test, test_targets['openhermes'], args.seed + 31, allow_repeat=False)
    test_records += sample_records(co_test, test_targets['coig_writer'], args.seed + 32, allow_repeat=False)
    test_records += sample_records(oa_test, test_targets['oasst1_zh'], args.seed + 33, allow_repeat=False)

    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    rng.shuffle(dev_records)
    rng.shuffle(test_records)

    train_issues = validate_final_records('train', train_records)
    dev_issues = validate_final_records('dev', dev_records)
    test_issues = validate_final_records('test', test_records)
    validation_issues = train_issues + dev_issues + test_issues

    write_jsonl(output_dir / 'train.jsonl', train_records)
    write_jsonl(output_dir / 'dev.jsonl', dev_records)
    write_jsonl(output_dir / 'test.jsonl', test_records)

    summary = {
        'target_total': args.target_total,
        'ratios': ratios,
        'effective_ratios': effective_ratios,
        'book_fallback_mode': book_fallback_mode,
        'train_targets': train_targets,
        'dev_targets': dev_targets,
        'test_targets': test_targets,
        'book_train_available': len(book_train),
        'openhermes_train_available': len(oh_train),
        'coig_train_available': len(co_train),
        'oasst1_train_available': len(oa_train),
        'token_filter': {
            'tokenizer_path': args.tokenizer_path,
            'book_train': book_train_filter_stats,
            'book_dev': book_dev_filter_stats,
            'book_test': book_test_filter_stats,
            'openhermes': openhermes_token_filter_stats,
            'coig_writer': coig_token_filter_stats,
            'oasst1_zh': oasst_token_filter_stats,
        },
        'openhermes_filter': openhermes_filter_stats,
        'train': summarize_split(train_records),
        'dev': summarize_split(dev_records),
        'test': summarize_split(test_records),
        'validation_issue_count': len(validation_issues),
        'validation_issue_examples': validation_issues[:20],
        'output_dir': str(output_dir),
    }
    (output_dir / 'build_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
