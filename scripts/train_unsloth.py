from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import unsloth  # noqa: F401
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from bookgpt.common import load_yaml


def load_model(config: dict):
    model_cfg = config['model']
    lora_cfg = config['lora']
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg['base_model_name'],
        max_seq_length=model_cfg['max_seq_length'],
        dtype=model_cfg.get('dtype'),
        load_in_4bit=model_cfg.get('load_in_4bit', True),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg['r'],
        target_modules=lora_cfg['target_modules'],
        lora_alpha=lora_cfg['alpha'],
        lora_dropout=lora_cfg['dropout'],
        bias=lora_cfg['bias'],
        use_gradient_checkpointing=lora_cfg.get('use_gradient_checkpointing', 'unsloth'),
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description='Train an Unsloth QLoRA SFT model.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_cfg = config['dataset']
    trainer_cfg = config['trainer']

    model, tokenizer = load_model(config)

    train_dataset = load_dataset('json', data_files=dataset_cfg['train_path'], split='train')
    do_eval = trainer_cfg.get('do_eval', False)
    eval_dataset = None
    if do_eval:
        eval_path = dataset_cfg.get('eval_path')
        if not eval_path:
            raise ValueError('dataset.eval_path is required when trainer.do_eval=true')
        eval_dataset = load_dataset('json', data_files=eval_path, split='train')

    sft_kwargs = dict(
        output_dir=trainer_cfg['output_dir'],
        per_device_train_batch_size=trainer_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=trainer_cfg['per_device_eval_batch_size'],
        gradient_accumulation_steps=trainer_cfg['gradient_accumulation_steps'],
        num_train_epochs=trainer_cfg.get('num_train_epochs', 1),
        learning_rate=trainer_cfg['learning_rate'],
        weight_decay=trainer_cfg['weight_decay'],
        logging_steps=trainer_cfg['logging_steps'],
        eval_steps=trainer_cfg.get('eval_steps'),
        save_steps=trainer_cfg['save_steps'],
        save_total_limit=trainer_cfg['save_total_limit'],
        lr_scheduler_type=trainer_cfg['lr_scheduler_type'],
        optim=trainer_cfg['optim'],
        bf16=trainer_cfg['bf16'],
        fp16=trainer_cfg['fp16'],
        gradient_checkpointing=trainer_cfg['gradient_checkpointing'],
        max_grad_norm=trainer_cfg['max_grad_norm'],
        report_to=trainer_cfg['report_to'],
        eval_strategy='steps' if do_eval else 'no',
        dataset_text_field=dataset_cfg['text_field'],
        max_length=config['model']['max_seq_length'],
        packing=False,
    )
    if 'max_steps' in trainer_cfg:
        sft_kwargs['max_steps'] = trainer_cfg['max_steps']
    if 'warmup_steps' in trainer_cfg:
        sft_kwargs['warmup_steps'] = trainer_cfg['warmup_steps']
    elif 'warmup_ratio' in trainer_cfg:
        sft_kwargs['warmup_ratio'] = trainer_cfg['warmup_ratio']

    training_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if do_eval else None,
        args=training_args,
    )

    resume_path = trainer_cfg.get('resume_from_checkpoint')
    if resume_path and Path(resume_path).exists():
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    output_dir = trainer_cfg['output_dir']
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    main()
