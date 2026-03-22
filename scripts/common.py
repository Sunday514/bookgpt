from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_tokenizer(tokenizer_path: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)


def count_text_tokens(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])


def annotate_token_counts(records: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for record in records:
        token_count = count_text_tokens(str(record.get('text', '')), tokenizer)
        updated = dict(record)
        meta = dict(updated.get('meta', {}))
        meta['token_count'] = token_count
        updated['meta'] = meta
        annotated.append(updated)
    return annotated


def is_peft_adapter_dir(model_path: str | Path) -> bool:
    path = Path(model_path)
    return (path / 'adapter_config.json').exists() and (path / 'adapter_model.safetensors').exists()


def load_hf_peft_4bit_model(model_path: str | Path, device_map: str = 'auto'):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path)
    adapter_config = json.loads((model_path / 'adapter_config.json').read_text(encoding='utf-8'))
    base_model_name_or_path = adapter_config['base_model_name_or_path']

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(model_path), is_trainable=False)
    return model, tokenizer


def load_causal_lm(model_path: str | Path, torch_dtype, device_map: str = 'auto'):
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    return model, tokenizer
