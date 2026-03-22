from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOK_INPUT = ROOT / 'dataset' / 'raw_books' / 'example_book.txt'
DEFAULT_BOOK_DIR = ROOT / 'dataset' / 'book_runs' / 'example_book'
DEFAULT_FINAL_DIR = ROOT / 'dataset' / 'final_runs' / 'smoke_1000_filtered'
DEFAULT_TOKENIZER = ROOT / 'models' / 'Qwen2.5-32B-Instruct-bnb-4bit'
DEFAULT_EXTERNAL = ROOT / 'dataset' / 'external_normalized'


def run(cmd: list[str]) -> None:
    print('+', ' '.join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the example smoke data pipeline end-to-end.')
    parser.add_argument('--book-input', default=str(DEFAULT_BOOK_INPUT))
    parser.add_argument('--book-dir', default=str(DEFAULT_BOOK_DIR))
    parser.add_argument('--final-dir', default=str(DEFAULT_FINAL_DIR))
    parser.add_argument('--external-dir', default=str(DEFAULT_EXTERNAL))
    parser.add_argument('--tokenizer-path', default=str(DEFAULT_TOKENIZER))
    parser.add_argument('--source-name', default='example_book_style')
    parser.add_argument('--target-total', type=int, default=1000)
    parser.add_argument('--ratios', default='0.45,0.20,0.25,0.10')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--min-tokens', type=int, default=32)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    python = sys.executable
    run([
        python,
        '-m',
        'bookgpt.data.build_book_dataset',
        '--input', args.book_input,
        '--output-dir', args.book_dir,
        '--source-name', args.source_name,
        '--min-chars', '320',
        '--max-chars', '900',
        '--train-variants', '4',
        '--eval-variants', '1',
        '--pivot-ratios', '0.42,0.50,0.58,0.66',
        '--seed', str(args.seed),
    ])
    run([
        python,
        '-m',
        'bookgpt.data.build_final_dataset',
        '--book-dir', args.book_dir,
        '--external-dir', args.external_dir,
        '--output-dir', args.final_dir,
        '--target-total', str(args.target_total),
        '--ratios', args.ratios,
        '--tokenizer-path', args.tokenizer_path,
        '--max-tokens', str(args.max_tokens),
        '--min-tokens', str(args.min_tokens),
        '--seed', str(args.seed),
    ])
    run([
        python,
        'scripts/inspect_dataset.py',
        '--dataset-dir', args.final_dir,
        '--tokenizer-path', args.tokenizer_path,
        '--write-summary',
    ])


if __name__ == '__main__':
    main()
