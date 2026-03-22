# 数据目录说明

这个目录只应长期保留两类内容：

- 小型示例输入
- 文档说明

其余内容大多是本地生成或下载产物，不应提交到 git。

## 目录用途

### `raw_books/`

原始中文 `.txt` 输入目录。

典型内容：

- `example_book.txt`：仓库内置的小示例
- `your_text.txt`：你自己的本地文本

除极小且可安全分发的示例外，真实书籍文本应保留在本地，不提交到仓库。

### `book_runs/`

`build_book_dataset.py` 或 `run_pipeline.py` 生成的书籍侧数据。

典型内容：

- `derived.jsonl`
- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`

这些文件是本地生成产物，不应提交。

### `final_runs/`

最终混合后的训练数据集。

典型内容：

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `build_summary.json`
- `dataset_summary.json`

这些文件也是本地生成产物，不应提交。

### `external_normalized/`

由 `scripts/prepare_external_datasets.py` 生成的归一化外部数据。

这些文件来自上游下载数据的本地加工结果，不应提交。

## 数据生成流程

1. 把上游数据下载到 `external_datasets/`
2. 运行 `scripts/prepare_external_datasets.py`
3. 把你自己的 `.txt` 放到 `dataset/raw_books/`
4. 运行 `scripts/run_pipeline.py`

## Git 规则

可以提交：

- 小型示例 `.txt`
- 这个目录下的说明文档

不要提交：

- 归一化外部数据
- 生成的 train/dev/test JSONL
- 私有或受限版权文本
- 从私有文本派生出的训练数据
