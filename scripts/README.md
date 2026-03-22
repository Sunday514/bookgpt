# 脚本目录说明

这个目录只保留对外公开的 CLI 入口。

## 推荐入口

- `prepare_external_datasets.py`
  - 归一化外部数据
- `run_pipeline.py`
  - 从中文 `.txt` 构建默认训练集
- `train_unsloth.py`
  - 启动默认训练
- `inspect_dataset.py`
  - 检查生成数据集
- `eval_model.py`
  - 跑离线生成评估
- `chat_local.py`
  - 本地推理对话
- `run_smoke_pipeline.py`
  - 运行示例 smoke 流程

内部实现代码已经迁移到 `bookgpt/` 包目录。
