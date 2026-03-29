from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path

import torch

from bookgpt.common import is_peft_adapter_dir, load_causal_lm, load_hf_peft_4bit_model


def load_presets(path: str | Path) -> dict:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def load_chat_model(model_path: str):
    if is_peft_adapter_dir(model_path):
        return load_hf_peft_4bit_model(model_path)
    return load_causal_lm(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Simple local Chinese writing chat loop.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--preset', default='continue_scene')
    parser.add_argument('--presets-file', default='serve/inference_presets.json')
    args = parser.parse_args()

    presets = load_presets(args.presets_file)
    preset = presets[args.preset]
    model, tokenizer = load_chat_model(args.model_path)

    system_prompt = '请直接、自然地回答用户问题；如果用户要求创作或改写，再按要求处理。'
    history = [{'role': 'system', 'content': system_prompt}]
    print('Enter `exit` to quit.')
    while True:
        user_text = input('\nuser> ').strip()
        if user_text.lower() in {'exit', 'quit'}:
            break
        history.append({'role': 'user', 'content': user_text})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=preset['max_new_tokens'],
            temperature=preset['temperature'],
            top_p=preset['top_p'],
            repetition_penalty=preset['repetition_penalty'],
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        history.append({'role': 'assistant', 'content': response})
        print(f'\nassistant> {response}')


if __name__ == '__main__':
    main()
