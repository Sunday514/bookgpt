from __future__ import annotations

import argparse

import torch

from common import load_causal_lm, read_jsonl, write_jsonl


def build_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Run offline generation eval over a JSONL prompt set.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=700)
    parser.add_argument('--temperature', type=float, default=0.85)
    parser.add_argument('--top-p', type=float, default=0.92)
    args = parser.parse_args()

    model, tokenizer = load_causal_lm(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
    )
    records = read_jsonl(args.prompts)
    results = []
    for record in records:
        prompt = build_prompt(tokenizer, record['messages'])
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
        )
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        results.append(
            {
                'id': record['id'],
                'category': record['category'],
                'messages': record['messages'],
                'response': generated.strip(),
            }
        )

    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} evaluation outputs to {args.output}")


if __name__ == '__main__':
    main()
