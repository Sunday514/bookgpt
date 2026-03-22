# Dataset Schema

All datasets use JSONL. Each line is one sample with a `messages` field and a `meta` field.

## Shape

```json
{
  "messages": [
    {"role": "system", "content": "你是中文小说创作助手。"},
    {"role": "user", "content": "任务描述..."},
    {"role": "assistant", "content": "目标输出..."}
  ],
  "meta": {
    "source": "novel_a_style|derived_scene|writing_task|general_instruct",
    "task_type": "continue|rewrite_pov|rewrite_tense|outline_to_prose|style_transfer|dialogue_enhance|qa",
    "style_tags": ["冷峻", "克制", "短对白", "环境描写强"],
    "split": "train|dev|test",
    "copyright_status": "licensed",
    "safety": {
      "memorization_holdout": false
    }
  }
}
```

## Rules

- `messages` order is fixed: `system`, `user`, `assistant`.
- `assistant` must contain the target answer.
- `meta` is for governance only and is not fed into the model directly.
- `style_tags` should be abstract. Do not encode book titles or author names.
- `memorization_holdout=true` samples are excluded from training and reserved for leakage checks.

## Recommended Dataset Mix

- `30%` style-conditioned original writing
- `25%` novel-derived transformation tasks
- `20%` writing workflow tasks
- `15%` general writing instruct tasks
- `10%` general instruct retention tasks
