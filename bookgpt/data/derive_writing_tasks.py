from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

from bookgpt.common import write_jsonl

CONTINUE_USER_PROMPTS = [
    '延续下面的情绪和叙事节奏续写，不要重复输入中的表述。\n\n{prompt_text}',
    '请接着下面的片段往后写，保持人物、场景和语气连贯，避免直接复述前文。\n\n{prompt_text}',
    '根据以下上文继续创作，维持原有文风与推进方式，不要照抄原句。\n\n{prompt_text}',
    '继续写下面这段内容，保持同一视角和氛围，自然推进情节，避免机械重复。\n\n{prompt_text}',
]

STYLE_TAG_HINTS = [
    '克制',
    '冷峻',
    '环境描写强',
    '对白短',
    '少解释',
    '压抑',
    '收束感强',
]

HEADING_PATTERNS = [
    re.compile(r'^第[0-9零一二三四五六七八九十百千万两〇]+[章节卷部集篇回幕].{0,30}$'),
    re.compile(r'^[0-9零一二三四五六七八九十百千万两〇]+[\.、]\s*.+$'),
    re.compile(r'^(序章|序幕|楔子|引子|前言|后记|尾声|终章|番外|附录)(\s*[:：].*)?$'),
    re.compile(r'^(chapter|part|prologue|epilogue)\b.*$', re.IGNORECASE),
]

TOC_PATTERNS = [
    re.compile(r'^目录$'),
    re.compile(r'^contents$', re.IGNORECASE),
    re.compile(r'^第[0-9零一二三四五六七八九十百千万两〇]+[章节卷部集篇回幕].*\d+$'),
    re.compile(r'^.+[\.·•]{3,}.+\d+$'),
]

DISCARD_PATTERNS = [
    re.compile(r'^www\.', re.IGNORECASE),
    re.compile(r'^https?://', re.IGNORECASE),
    re.compile(r'^(手机访问|最新网址|下载地址|版权归原作者所有|本书来自|仅供试读|扫码下载).*$'),
]


def read_text(path: str | Path) -> str:
    path = Path(path)
    for encoding in ('utf-8', 'utf-8-sig', 'gb18030', 'gbk'):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding='utf-8', errors='ignore')


def clean_text(text: str) -> str:
    text = text.replace('\ufeff', '')
    text = text.replace('\u3000', ' ')
    text = text.replace('\xa0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'[ \f\v]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r'\s+', ' ', line)
    return line


def is_probable_heading(line: str) -> bool:
    if not line or len(line) > 60:
        return False
    return any(pattern.match(line) for pattern in HEADING_PATTERNS)


def is_probable_toc_line(line: str) -> bool:
    if not line or len(line) > 80:
        return False
    return any(pattern.match(line) for pattern in TOC_PATTERNS) or (
        is_probable_heading(line) and bool(re.search(r'\d+$', line))
    )


def is_discardable_line(line: str) -> bool:
    if not line:
        return False
    return any(pattern.match(line) for pattern in DISCARD_PATTERNS)


def strip_front_matter(lines: list[str]) -> tuple[list[str], dict[str, int]]:
    non_empty = [line for line in lines if line]
    scan_window = non_empty[:200]
    candidate_indexes = [idx for idx, line in enumerate(scan_window) if is_probable_toc_line(line)]

    if len(candidate_indexes) < 5 or candidate_indexes[0] > 12:
        return lines, {'front_matter_lines_removed': 0}

    cut_non_empty = candidate_indexes[-1] + 1
    removed = 0
    kept_non_empty = 0
    stripped: list[str] = []

    for line in lines:
        if not line:
            if kept_non_empty == 0:
                removed += 1
                continue
            stripped.append(line)
            continue
        if kept_non_empty < cut_non_empty:
            kept_non_empty += 1
            removed += 1
            continue
        stripped.append(line)

    return stripped, {'front_matter_lines_removed': removed}


def join_lines(lines: list[str]) -> str:
    text = ''
    for line in lines:
        if not text:
            text = line
            continue
        prev = text[-1]
        curr = line[0]
        if prev.isascii() and prev.isalnum() and curr.isascii() and curr.isalnum():
            text += ' ' + line
        else:
            text += line
    return text.strip()


def extract_book_paragraphs(text: str) -> tuple[list[str], dict[str, int]]:
    cleaned = clean_text(text)
    raw_lines = [normalize_line(line) for line in cleaned.split('\n')]
    raw_lines, strip_stats = strip_front_matter(raw_lines)

    paragraphs: list[str] = []
    current_lines: list[str] = []
    pending_heading: str | None = None
    chapter_headings = 0
    discarded_lines = 0

    def flush_current() -> None:
        nonlocal current_lines, pending_heading
        if not current_lines:
            return
        paragraph = join_lines(current_lines)
        current_lines = []
        if pending_heading:
            paragraph = f'{pending_heading}\n{paragraph}'
            pending_heading = None
        if len(paragraph) >= 20:
            paragraphs.append(paragraph)

    for line in raw_lines:
        if not line:
            flush_current()
            continue
        if is_discardable_line(line):
            discarded_lines += 1
            continue
        if is_probable_heading(line):
            flush_current()
            pending_heading = line
            chapter_headings += 1
            continue
        current_lines.append(line)

    flush_current()

    stats = {
        'raw_line_count': len(raw_lines),
        'paragraph_count': len(paragraphs),
        'chapter_heading_count': chapter_headings,
        'discarded_line_count': discarded_lines,
        **strip_stats,
    }
    return paragraphs, stats


def chunk_paragraphs(paragraphs: list[str], min_chars: int, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        if current and current_len + para_len > max_chars:
            chunk = '\n\n'.join(current).strip()
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len
    if current:
        chunk = '\n\n'.join(current).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
    return chunks


def chunk_text_fallback(text: str, min_chars: int, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []

    chunks: list[str] = []
    start = 0
    step = max(min_chars, int(max_chars * 0.75))
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def infer_style_tags(chunk: str) -> list[str]:
    tags: list[str] = []
    if '“' in chunk or '”' in chunk:
        tags.append('对白短')
    if any(word in chunk for word in ['雨', '风', '灯', '夜', '门', '窗']):
        tags.append('环境描写强')
    if any(word in chunk for word in ['没有', '只是', '仍然', '终于']):
        tags.append('克制')
    if any(word in chunk for word in ['冷', '暗', '灰', '空', '静']):
        tags.append('冷峻')
    if not tags:
        tags = random.sample(STYLE_TAG_HINTS, 3)
    return tags[:4]


def parse_pivot_ratios(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        ratios = [float(item.strip()) for item in value.split(',') if item.strip()]
    else:
        ratios = [float(item) for item in value]
    valid = [ratio for ratio in ratios if 0.20 <= ratio <= 0.80]
    if not valid:
        raise ValueError('At least one pivot ratio in [0.20, 0.80] is required.')
    return valid


def collect_boundary_candidates(chunk: str) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    sentence_endings = "。！？!?；;…"
    closing_chars = "”』」）】\""

    for index, char in enumerate(chunk):
        if char in sentence_endings:
            boundary = index + 1
            while boundary < len(chunk) and chunk[boundary] in closing_chars:
                boundary += 1
            candidates.append((boundary, 'sentence'))
        elif char == "\n":
            boundary = index + 1
            if boundary < len(chunk) and chunk[boundary] == "\n":
                candidates.append((boundary + 1, 'paragraph'))

    deduped: list[tuple[int, str]] = []
    seen: set[int] = set()
    for position, kind in candidates:
        if position in seen:
            continue
        seen.add(position)
        deduped.append((position, kind))
    return deduped

def choose_continue_pivot(chunk: str, pivot_ratio: float) -> tuple[int, str]:
    min_prompt = 120
    min_target = 80
    ideal = int(len(chunk) * pivot_ratio)
    min_pivot = min_prompt
    max_pivot = len(chunk) - min_target
    ideal = max(min_pivot, min(max_pivot, ideal))

    candidates = [
        (position, kind)
        for position, kind in collect_boundary_candidates(chunk)
        if min_pivot <= position <= max_pivot
    ]
    if not candidates:
        return ideal, 'char'

    preferred_window = max(40, min(160, len(chunk) // 10))
    sentence_candidates = [
        (position, kind)
        for position, kind in candidates
        if kind == 'sentence' and abs(position - ideal) <= preferred_window
    ]
    if sentence_candidates:
        best = min(sentence_candidates, key=lambda item: abs(item[0] - ideal))
        return best

    paragraph_candidates = [
        (position, kind)
        for position, kind in candidates
        if kind == 'paragraph' and abs(position - ideal) <= preferred_window
    ]
    if paragraph_candidates:
        best = min(paragraph_candidates, key=lambda item: abs(item[0] - ideal))
        return best

    best = min(candidates, key=lambda item: abs(item[0] - ideal))
    return best


def build_continue_sample(
    chunk: str,
    source_name: str,
    split: str,
    *,
    pivot_ratio: float = 0.5,
    template_index: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, object] | None:
    if len(chunk) < 240:
        return None

    pivot, boundary_kind = choose_continue_pivot(chunk, pivot_ratio)
    prompt_text = chunk[:pivot].strip()
    target_text = chunk[pivot:].strip()
    if len(target_text) < 80 or len(prompt_text) < 120:
        return None

    if rng is None:
        rng = random

    if template_index is None:
        template_index = rng.randrange(len(CONTINUE_USER_PROMPTS))

    user_prompt = CONTINUE_USER_PROMPTS[template_index].format(prompt_text=prompt_text)
    return {
        'messages': [
            {'role': 'user', 'content': user_prompt},
            {'role': 'assistant', 'content': target_text},
        ],
        'meta': {
            'source': source_name,
            'task_type': 'continue',
            'style_tags': infer_style_tags(chunk),
            'split': split,
            'copyright_status': 'licensed',
            'prompt_template': f'continue_u{template_index}',
            'pivot_ratio': round(pivot_ratio, 4),
            'pivot_boundary': boundary_kind,
            'safety': {'memorization_holdout': split == 'test'},
        },
    }


def generate_continue_variants(
    chunk: str,
    source_name: str,
    split: str,
    *,
    variant_count: int,
    pivot_ratios: list[float],
    rng: random.Random,
) -> list[dict[str, object]]:
    if variant_count <= 0:
        return []

    records: list[dict[str, object]] = []
    for variant_index in range(variant_count):
        pivot_ratio = pivot_ratios[variant_index % len(pivot_ratios)]
        template_index = rng.randrange(len(CONTINUE_USER_PROMPTS))
        record = build_continue_sample(
            chunk,
            source_name,
            split,
            pivot_ratio=pivot_ratio,
            template_index=template_index,
            rng=rng,
        )
        if record is None:
            continue
        record['meta']['variant_index'] = variant_index
        records.append(record)
    return records


def assign_split(index: int, total: int) -> str:
    if total <= 1:
        return 'train'
    if total == 2:
        return 'train' if index == 0 else 'test'
    if total == 3:
        if index == 0:
            return 'train'
        if index == 1:
            return 'dev'
        return 'test'
    if index >= max(total - 1, 0):
        return 'test'
    if index >= max(total - 2, 0):
        return 'dev'
    return 'train'


def derive_records(
    chunks: list[str],
    source_name: str,
    *,
    train_variants: int,
    eval_variants: int,
    pivot_ratios: list[float],
    seed: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        split = assign_split(idx, total)
        variant_count = train_variants if split == 'train' else eval_variants
        rng = random.Random(seed + idx)
        records.extend(
            generate_continue_variants(
                chunk,
                source_name,
                split,
                variant_count=variant_count,
                pivot_ratios=pivot_ratios,
                rng=rng,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description='Derive continue-task JSONL from a licensed book text.')
    parser.add_argument('--input', required=True, help='Path to raw book text file.')
    parser.add_argument('--output', required=True, help='Path to output JSONL.')
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
    records = derive_records(
        chunks,
        args.source_name,
        train_variants=args.train_variants,
        eval_variants=args.eval_variants,
        pivot_ratios=pivot_ratios,
        seed=args.seed,
    )
    write_jsonl(args.output, records)

    summary = {
        **parse_stats,
        'chunks': len(chunks),
        'records': len(records),
        'train_variants': args.train_variants,
        'eval_variants': args.eval_variants,
        'pivot_ratios': pivot_ratios,
        'output': args.output,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
