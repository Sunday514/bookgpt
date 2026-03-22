from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOK_DIR = ROOT / 'dataset' / 'book_runs' / 'current'
DEFAULT_FINAL_DIR = ROOT / 'dataset' / 'final_runs' / 'current'
DEFAULT_TOKENIZER = ROOT / 'models' / 'Qwen2.5-32B-Instruct-bnb-4bit'
DEFAULT_EXTERNAL = ROOT / 'dataset' / 'external_normalized'


def run(cmd: list[str]) -> None:
    print('+', ' '.join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the default training dataset from any book txt input.')
    parser.add_argument('--book-input', required=True, help='Path to any book-style txt file.')
    parser.add_argument('--book-dir', default=str(DEFAULT_BOOK_DIR), help='Book-derived dataset output directory.')
    parser.add_argument('--final-dir', default=str(DEFAULT_FINAL_DIR), help='Final mixed dataset output directory.')
    parser.add_argument('--external-dir', default=str(DEFAULT_EXTERNAL), help='Normalized external dataset directory.')
    parser.add_argument('--tokenizer-path', default=str(DEFAULT_TOKENIZER), help='Tokenizer path used for token filtering.')
    parser.add_argument('--source-name', help='Optional source name. Defaults to <txt_stem>_style.')
    parser.add_argument('--target-total', type=int, default=10000)
    parser.add_argument('--ratios', default='0.45,0.20,0.25,0.10')
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--min-tokens', type=int, default=32)
    parser.add_argument('--min-chars', type=int, default=320)
    parser.add_argument('--max-chars', type=int, default=900)
    parser.add_argument('--train-variants', type=int, default=4)
    parser.add_argument('--eval-variants', type=int, default=1)
    parser.add_argument('--pivot-ratios', default='0.42,0.50,0.58,0.66')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    book_input = Path(args.book_input).expanduser().resolve()
    source_name = args.source_name or f'{book_input.stem}_style'
    python = sys.executable

    run([
        python,
        'scripts/build_book_dataset.py',
        '--input', str(book_input),
        '--output-dir', args.book_dir,
        '--source-name', source_name,
        '--min-chars', str(args.min_chars),
        '--max-chars', str(args.max_chars),
        '--train-variants', str(args.train_variants),
        '--eval-variants', str(args.eval_variants),
        '--pivot-ratios', args.pivot_ratios,
        '--seed', str(args.seed),
    ])
    run([
        python,
        'scripts/build_final_dataset.py',
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
