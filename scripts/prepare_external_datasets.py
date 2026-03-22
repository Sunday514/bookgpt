from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any

from common import write_jsonl

DEFAULT_OPENHERMES_PATH = Path('external_datasets/OpenHermes-2.5-zh/translation_filtered.jsonl')
DEFAULT_OASST1_PATH = Path('external_datasets/oasst1/data/train-00000-of-00001-b42a775f407cee45.parquet')
DEFAULT_COIG_WRITER_PATH = Path('external_datasets/COIG-Writer/all_human_data.json')


def build_formatted_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message['role']
        content = message['content'].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return '\n'.join(parts)


def normalize_content(text: str | None) -> str:
    if text is None:
        return ''
    return str(text).replace('\r\n', '\n').replace('\r', '\n').strip()


def finalize_record(messages: list[dict[str, str]], meta: dict[str, Any]) -> dict[str, Any] | None:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get('role')
        content = normalize_content(message.get('content'))
        if role not in {'system', 'user', 'assistant'} or not content:
            continue
        normalized.append({'role': role, 'content': content})

    if len(normalized) < 2 or normalized[-1]['role'] != 'assistant':
        return None

    record = {
        'messages': normalized,
        'meta': meta,
    }
    record['text'] = build_formatted_text(normalized)
    return record


def convert_openhermes(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    role_map = {'system': 'system', 'human': 'user', 'gpt': 'assistant'}
    records: list[dict[str, Any]] = []

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            raw = json.loads(line)
            conversations = raw.get('conversations') or []
            messages: list[dict[str, str]] = []
            for turn in conversations:
                role = role_map.get(turn.get('from'))
                content = normalize_content(turn.get('value'))
                if role is None or not content:
                    continue
                messages.append({'role': role, 'content': content})

            record = finalize_record(
                messages,
                {
                    'source': 'openhermes_2_5_zh',
                    'task_type': 'general_instruct',
                    'language': 'zh',
                    'copyright_status': 'open',
                    'license': 'apache-2.0',
                    'original_source': raw.get('source'),
                },
            )
            if record is None:
                continue
            records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break

    return records


def get_label_value(labels: dict[str, Any] | None, name: str) -> float | None:
    if not labels:
        return None
    label_names = labels.get('name') or []
    label_values = labels.get('value') or []
    for idx, label_name in enumerate(label_names):
        if label_name == name and idx < len(label_values):
            return label_values[idx]
    return None


def build_oasst_messages(node_id: str, by_id: dict[str, dict[str, Any]]) -> list[dict[str, str]] | None:
    chain: list[dict[str, Any]] = []
    current = by_id.get(node_id)
    visited: set[str] = set()

    while current is not None:
        message_id = current['message_id']
        if message_id in visited:
            return None
        visited.add(message_id)
        chain.append(current)
        parent_id = current.get('parent_id')
        current = by_id.get(parent_id) if parent_id else None

    chain.reverse()
    role_map = {'prompter': 'user', 'assistant': 'assistant'}
    messages: list[dict[str, str]] = []
    for node in chain:
        role = role_map.get(node.get('role'))
        content = normalize_content(node.get('text'))
        if role is None or not content:
            return None
        if messages and messages[-1]['role'] == role:
            return None
        messages.append({'role': role, 'content': content})

    if not messages or messages[0]['role'] != 'user' or messages[-1]['role'] != 'assistant':
        return None
    return messages


def convert_oasst1(path: Path, max_samples: int | None = None, min_quality: float = 0.7) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows = table.to_pylist()
    by_id = {row['message_id']: row for row in rows}

    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get('role') != 'assistant':
            continue
        if row.get('deleted'):
            continue
        if not row.get('review_result'):
            continue
        lang = str(row.get('lang') or '')
        if not lang.startswith('zh'):
            continue
        quality = get_label_value(row.get('labels'), 'quality')
        if quality is not None and quality < min_quality:
            continue

        messages = build_oasst_messages(row['message_id'], by_id)
        if messages is None:
            continue

        record = finalize_record(
            messages,
            {
                'source': 'oasst1',
                'task_type': 'chat',
                'language': lang,
                'copyright_status': 'open',
                'license': 'apache-2.0',
                'quality_score': quality,
                'message_tree_id': row.get('message_tree_id'),
            },
        )
        if record is None:
            continue
        records.append(record)
        if max_samples is not None and len(records) >= max_samples:
            break

    return records


def convert_coig_writer(path: Path, max_samples: int | None = None, include_thought: bool = False) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    records: list[dict[str, Any]] = []

    for raw in data:
        query = normalize_content(raw.get('query'))
        answer = normalize_content(raw.get('answer'))
        thought = normalize_content(raw.get('thought'))
        if not query or not answer:
            continue

        if include_thought and thought:
            assistant = f"写作思路：\n{thought}\n\n正文：\n{answer}"
            task_type = 'creative_writing_with_reasoning'
        else:
            assistant = answer
            task_type = 'creative_writing'

        record = finalize_record(
            [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': assistant},
            ],
            {
                'source': 'coig_writer',
                'task_type': task_type,
                'language': 'zh',
                'copyright_status': 'open',
                'license': 'odc-by',
                'record_id': raw.get('id'),
            },
        )
        if record is None:
            continue
        records.append(record)
        if max_samples is not None and len(records) >= max_samples:
            break

    return records


def write_outputs(output_dir: Path, named_records: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, Any]] = []
    summary: dict[str, Any] = {'output_dir': str(output_dir), 'sources': {}}

    for name, records in named_records.items():
        write_jsonl(output_dir / f'{name}.jsonl', records)
        combined.extend(records)
        summary['sources'][name] = {'records': len(records)}

    rng = random.Random(seed)
    rng.shuffle(combined)
    write_jsonl(output_dir / 'external_all.jsonl', combined)
    summary['combined_records'] = len(combined)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert external datasets into unified Qwen SFT JSONL format.')
    parser.add_argument('--output-dir', required=True, help='Directory for normalized JSONL files.')
    parser.add_argument('--openhermes-path', default=str(DEFAULT_OPENHERMES_PATH))
    parser.add_argument('--oasst1-path', default=str(DEFAULT_OASST1_PATH))
    parser.add_argument('--coig-writer-path', default=str(DEFAULT_COIG_WRITER_PATH))
    parser.add_argument('--max-openhermes', type=int, default=None)
    parser.add_argument('--max-oasst1', type=int, default=None)
    parser.add_argument('--max-coig-writer', type=int, default=None)
    parser.add_argument('--oasst1-min-quality', type=float, default=0.7)
    parser.add_argument('--include-coig-thought', action='store_true')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    openhermes_records = convert_openhermes(Path(args.openhermes_path), args.max_openhermes)
    oasst1_records = convert_oasst1(Path(args.oasst1_path), args.max_oasst1, args.oasst1_min_quality)
    coig_writer_records = convert_coig_writer(
        Path(args.coig_writer_path),
        args.max_coig_writer,
        include_thought=args.include_coig_thought,
    )

    summary = write_outputs(
        Path(args.output_dir),
        {
            'openhermes_2_5_zh': openhermes_records,
            'oasst1_zh': oasst1_records,
            'coig_writer': coig_writer_records,
        },
        args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
