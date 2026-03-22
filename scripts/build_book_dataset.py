from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import write_jsonl
from derive_writing_tasks import (
    chunk_paragraphs,
    chunk_text_fallback,
    derive_records,
    extract_book_paragraphs,
    parse_pivot_ratios,
    read_text,
)


def build_formatted_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message['role']
        content = message['content'].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return '\n'.join(parts)


def finalize_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train: list[dict] = []
    dev: list[dict] = []
    test: list[dict] = []

    for record in records:
        record['text'] = build_formatted_text(record['messages'])
        split = record.get('meta', {}).get('split', 'train')
        holdout = record.get('meta', {}).get('safety', {}).get('memorization_holdout', False)

        if split == 'train' and holdout:
            split = 'test'
            record['meta']['split'] = 'test'

        if split == 'train':
            train.append(record)
        elif split == 'dev':
            dev.append(record)
        else:
            test.append(record)

    return train, dev, test


def summarize(
    records: list[dict],
    chunks: list[str],
    output_dir: Path,
    parse_stats: dict[str, int],
    *,
    fallback_used: bool,
) -> dict:
    split_counter = Counter(record['meta'].get('split', 'unknown') for record in records)
    task_counter = Counter(record['meta'].get('task_type', 'unknown') for record in records)
    holdout_count = sum(
        1 for record in records if record.get('meta', {}).get('safety', {}).get('memorization_holdout', False)
    )
    template_counter = Counter(record['meta'].get('prompt_template', 'unknown') for record in records)
    pivot_counter = Counter(str(record['meta'].get('pivot_ratio', 'unknown')) for record in records)
    return {
        **parse_stats,
        'chunks': len(chunks),
        'records': len(records),
        'split_distribution': dict(split_counter),
        'task_distribution': dict(task_counter),
        'prompt_templates': dict(template_counter),
        'pivot_ratios': dict(pivot_counter),
        'memorization_holdout_records': holdout_count,
        'fallback_chunking_used': fallback_used,
        'output_dir': str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Build continue-only train/dev/test JSONL files from a book txt file.')
    parser.add_argument('--input', required=True, help='Path to a raw book txt file.')
    parser.add_argument('--output-dir', required=True, help='Directory for derived/train/dev/test files.')
    parser.add_argument('--source-name', default='book_a_style')
    parser.add_argument('--min-chars', type=int, default=300)
    parser.add_argument('--max-chars', type=int, default=1200)
    parser.add_argument('--train-variants', type=int, default=4)
    parser.add_argument('--eval-variants', type=int, default=1)
    parser.add_argument('--pivot-ratios', default='0.42,0.50,0.58,0.66')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    pivot_ratios = parse_pivot_ratios(args.pivot_ratios)
    raw = read_text(args.input)
    paragraphs, parse_stats = extract_book_paragraphs(raw)
    chunks = chunk_paragraphs(paragraphs, args.min_chars, args.max_chars)
    fallback_used = False
    if not chunks:
        relaxed_min = max(120, min(args.min_chars, 240))
        relaxed_max = max(args.max_chars, 600)
        chunks = chunk_paragraphs(paragraphs, relaxed_min, relaxed_max)
        if not chunks:
            combined_text = '\n\n'.join(paragraphs).strip()
            chunks = chunk_text_fallback(combined_text, min_chars=240, max_chars=relaxed_max)
        fallback_used = bool(chunks)
    records = derive_records(
        chunks,
        args.source_name,
        train_variants=args.train_variants,
        eval_variants=args.eval_variants,
        pivot_ratios=pivot_ratios,
        seed=args.seed,
    )
    train, dev, test = finalize_records(records)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / 'derived.jsonl', records)
    write_jsonl(output_dir / 'train.jsonl', train)
    write_jsonl(output_dir / 'dev.jsonl', dev)
    write_jsonl(output_dir / 'test.jsonl', test)

    summary = summarize(records, chunks, output_dir, parse_stats, fallback_used=fallback_used)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
